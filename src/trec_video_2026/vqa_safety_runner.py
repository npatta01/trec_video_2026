"""Resumable two-pass Qwen3.5 runner for the private 2026 AG safety run."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_models import build_controlled_messages
from trec_video_2026.vqa_submission import canonical_sha256
from trec_video_2026.vqa_safety_submission import (
    build_prediction_row,
    build_ranked_repair_messages,
    build_ranked_tail_messages,
    validate_resume_prefix,
)
from trec_video_2026.vqa_transformers_runner import (
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    generate_from_messages,
    load_processor,
    validate_bfloat16_accelerator,
)


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tail-max-new-tokens", type=int, default=512)
    parser.add_argument("--repair-max-new-tokens", type=int, default=256)
    return parser.parse_args()


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "run_id": run_id, "predictions": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise ValueError("existing checkpoint run identity mismatch")
    if not isinstance(payload.get("predictions"), list):
        raise ValueError("existing checkpoint predictions are invalid")
    return payload


def main() -> int:
    args = parse_args()
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        requests = requests[: args.limit]

    checkpoint = _load_checkpoint(args.output, args.run_id)
    start_index = validate_resume_prefix(requests, checkpoint["predictions"])
    if start_index == len(requests):
        print(json.dumps({"status": "complete", "rows": start_index}, sort_keys=True))
        return 0

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_started = time.monotonic()
    processor = load_processor(
        AutoProcessor,
        MODEL_ID,
        MODEL_REVISION,
        min_pixels=MIN_PIXELS_PER_FRAME,
        max_pixels=MAX_PIXELS_PER_FRAME,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    device, precision, attestation = validate_bfloat16_accelerator(model, torch)
    runtime = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "precision": precision,
        "runtime_attestation": attestation,
        "runtime_image": os.environ.get("VQA_RUNTIME_IMAGE"),
        "model_load_seconds": time.monotonic() - load_started,
    }
    checkpoint.update(runtime)

    for request in requests[start_index:]:
        frame_paths = tuple(Path(value) for value in request["frame_paths"])
        frame_timestamps = tuple(float(value) for value in request["frame_timestamps"])

        def generate_primary(row: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
            messages = build_controlled_messages(
                question=row["question"],
                frame_paths=frame_paths,
                frame_timestamps=frame_timestamps,
                transcript=None,
                answer_policy="current",
            )
            started = time.monotonic()
            try:
                answer = generate_from_messages(
                    processor=processor,
                    model=model,
                    messages=messages,
                    max_new_tokens=64,
                    normalize=True,
                )
                status = "passed"
                error = None
            except Exception as exc:  # noqa: BLE001 - preserve complete safety rows
                answer = "The answer cannot be determined"
                status = "fallback"
                error = f"{type(exc).__name__}: {str(exc)[:1000]}"
            return (
                answer,
                time.monotonic() - started,
                {
                    "status": status,
                    "error": error,
                    "request_sha256": canonical_sha256(messages),
                    **runtime,
                },
            )

        def generate_tail(
            row: dict[str, Any], primary: str
        ) -> tuple[str, float, dict[str, Any]]:
            messages = build_ranked_tail_messages(
                question=row["question"],
                frame_paths=frame_paths,
                frame_timestamps=frame_timestamps,
                primary_answer=primary,
            )
            started = time.monotonic()
            try:
                text = generate_from_messages(
                    processor=processor,
                    model=model,
                    messages=messages,
                    max_new_tokens=args.tail_max_new_tokens,
                    normalize=False,
                )
                status = "passed"
                error = None
            except Exception as exc:  # noqa: BLE001 - preserve complete safety rows
                text = "{}"
                status = "fallback"
                error = f"{type(exc).__name__}: {str(exc)[:1000]}"
            return (
                text,
                time.monotonic() - started,
                {
                    "status": status,
                    "error": error,
                    "request_sha256": canonical_sha256(messages),
                    **runtime,
                },
            )

        def generate_repair(
            row: dict[str, Any], excluded: tuple[str, ...], missing: int
        ) -> tuple[str, float, dict[str, Any]]:
            messages = build_ranked_repair_messages(
                question=row["question"],
                frame_paths=frame_paths,
                frame_timestamps=frame_timestamps,
                excluded_answers=excluded,
                missing_count=missing,
            )
            started = time.monotonic()
            try:
                text = generate_from_messages(
                    processor=processor,
                    model=model,
                    messages=messages,
                    max_new_tokens=args.repair_max_new_tokens,
                    normalize=False,
                )
                status = "passed"
                error = None
            except Exception as exc:  # noqa: BLE001 - preserve complete safety rows
                text = "{}"
                status = "fallback"
                error = f"{type(exc).__name__}: {str(exc)[:1000]}"
            return (
                text,
                time.monotonic() - started,
                {
                    "status": status,
                    "error": error,
                    "request_sha256": canonical_sha256(messages),
                    **runtime,
                },
            )

        row = build_prediction_row(
            request,
            generate_primary=generate_primary,
            generate_tail=generate_tail,
            generate_repair=generate_repair,
        )
        checkpoint["predictions"].append(row)
        _atomic_checkpoint(args.output, checkpoint)
        print(
            json.dumps(
                {
                    "completed": len(checkpoint["predictions"]),
                    "total": len(requests),
                    "query_id": request["query_id"],
                    "fallback_count": row["fallback_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
