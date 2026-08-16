#!/usr/bin/env python3
"""Run the fixed, immutable Gemini 3.7 Flash two-video smoke transitions."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from google import genai

import trec_video_2026.vqa_gemini37_flash_smoke as smoke


_REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE_ROOT = _REPO_ROOT / "data/vqa/benchmarks/direct-gemini-batch-475-v1"
PRIVATE_RUN_ROOT = _REPO_ROOT / "data/vqa/benchmarks/gemini37-flash-two-video-smoke-v1"
CANONICAL_ANNOTATIONS_ROOT = _REPO_ROOT / "data/vqa/2025/training/annotations"


def client_factory(*, api_key: str) -> Any:
    """Construct the provider client only for explicitly live commands."""

    return genai.Client(api_key=api_key)


def _parser() -> argparse.ArgumentParser:
    class Parser(argparse.ArgumentParser):
        def error(self, message: str) -> None:
            raise ValueError("invalid command")

    parser = Parser(
        prog="run_gemini37_flash_smoke",
        description=__doc__,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight", "upload", "freeze", "score", "show-status"):
        commands.add_parser(name)
    for name in ("submit", "status"):
        command = commands.add_parser(name)
        command.add_argument("--arm", required=True, choices=tuple(smoke.ARMS))
    return parser


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _aggregate_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    rows = payload.get("rows")
    if isinstance(rows, list):
        counts["rows"] = len(rows)
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict) and isinstance(aggregate.get("row_count"), int):
        counts["rows"] = aggregate["row_count"]
    arms = payload.get("arms")
    if isinstance(arms, dict):
        counts["arms"] = len(arms)
        response_count = 0
        for value in arms.values():
            if isinstance(value, dict) and isinstance(value.get("response_count"), int):
                response_count += value["response_count"]
        if response_count:
            counts["responses"] = response_count
    for source, target in (
        ("active_count", "active"),
        ("response_count", "responses"),
        ("request_count", "requests"),
        ("rows_verified", "rows_verified"),
    ):
        value = payload.get(source)
        if isinstance(value, int):
            counts[target] = value
    paired = aggregate.get("paired") if isinstance(aggregate, dict) else None
    if isinstance(paired, dict):
        for source, target in (
            ("candidate_wins", "candidate_wins"),
            ("ties", "ties"),
            ("candidate_losses", "candidate_losses"),
        ):
            value = paired.get(source)
            if isinstance(value, int):
                counts[target] = value
    return counts


def _emit(payload: dict[str, Any]) -> None:
    """Print only aggregate-safe fields; all row/provider fields remain private."""

    output: dict[str, Any] = {"status": payload.get("status")}
    arm = payload.get("arm")
    if isinstance(arm, str) and arm in smoke.ARMS:
        output["arm"] = arm
    counts = _aggregate_counts(payload)
    if counts:
        output["counts"] = counts
    for key in ("decision", "estimated_cost_ceiling_usd", "observed_cost_usd"):
        value = payload.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            output[key] = value
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


def _emit_error() -> None:
    print(json.dumps({"status": "error", "error": "command failed"}, separators=(",", ":")))


def _client(factory: Callable[..., Any]) -> Any:
    try:
        key = os.environ["GEMINI_API_KEY"]
    except KeyError as error:
        raise ValueError("GEMINI_API_KEY is not configured") from error
    return factory(api_key=key)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> int:
    """Dispatch one fixed transition and reduce its result to safe aggregates."""

    try:
        args = _parser().parse_args(argv)
        factory = client_factory or globals()["client_factory"]
        command = args.command
        if command == "prepare":
            result = smoke.prepare_protocol(
                source_root=CANONICAL_SOURCE_ROOT,
                root=PRIVATE_RUN_ROOT,
                created_at=_now(),
            )
        elif command == "preflight":
            result = smoke.preflight_models(PRIVATE_RUN_ROOT, _client(factory), _now())
        elif command == "upload":
            result = smoke.upload_rows(PRIVATE_RUN_ROOT, _client(factory), sleep=time.sleep)
        elif command == "submit":
            result = smoke.submit_arm(
                PRIVATE_RUN_ROOT,
                args.arm,
                _client(factory),
                submitted_at=_now(),
            )
        elif command == "status":
            result = smoke.record_status(
                PRIVATE_RUN_ROOT,
                args.arm,
                _client(factory),
                checked_at=_now(),
            )
        elif command == "freeze":
            result = smoke.freeze_predictions(PRIVATE_RUN_ROOT, frozen_at=_now())
        elif command == "score":
            result = smoke.score_postfreeze(
                PRIVATE_RUN_ROOT,
                CANONICAL_ANNOTATIONS_ROOT,
                scored_at=_now(),
            )
        elif command == "show-status":
            result = smoke.show_status(PRIVATE_RUN_ROOT)
        else:  # pragma: no cover - argparse choices make this unreachable
            raise ValueError("invalid command")
        _emit(result)
        return 0
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - CLI boundary must not leak private errors
        _emit_error()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
