#!/usr/bin/env python3
"""Safe command-line adapter for the immutable Phase 0 runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from trec_video_2026.vqa_frontier_phase0_runner import (
    ANSWER_POLICY,
    MAP_POLICY,
    AmbiguousStartedAttemptError,
    CostGateError,
    PilotGateError,
    Phase0RunSummary,
    Phase0RunnerError,
    authorize_remaining,
    authorize_screen,
    freeze_pilot,
    freeze_screen,
    load_prepared,
    preflight,
    prepare_run,
    project_remaining_cost,
    run_pilot,
    run_screen_remaining,
    run_screen_row,
    status,
)
from trec_video_2026.vqa_submission_openrouter import OpenRouterTransport


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        payload = _dispatch(args)
    except Exception as error:
        print(json.dumps({"status": "error", "error": _safe_error(error)}), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_frontier_phase0.py")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--panel", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    for name in ("preflight", "run-pilot", "freeze-pilot", "project-cost", "authorize-remaining", "freeze-screen", "status", "run-screen-remaining"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)

    authorize = commands.add_parser("authorize-screen")
    authorize.add_argument("--prepared", type=Path, required=True)
    authorize.add_argument("--pilot-audit-freeze", type=Path, required=True)

    row = commands.add_parser("run-screen-row")
    row.add_argument("--prepared", type=Path, required=True)
    row.add_argument("--ordinal", type=int, choices=range(1, 8), required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "prepare":
        prepared = prepare_run(args.panel, args.output)
        return {
            "status": "prepared",
            "run_id": prepared.run_id,
            "panel_sha256": prepared.panel_sha256,
            "pilot_count": 1,
            "screen_count": len(prepared.screen_rows),
        }
    prepared = load_prepared(args.prepared)
    if args.command == "preflight":
        return dict(preflight(prepared))
    if args.command == "status":
        return dict(status(prepared))
    if args.command == "project-cost":
        return dict(project_remaining_cost(prepared))
    if args.command == "run-pilot":
        return _summary(run_pilot(prepared, _transport("neutral_map")))
    if args.command == "freeze-pilot":
        return dict(freeze_pilot(prepared))
    if args.command == "authorize-screen":
        return dict(authorize_screen(prepared, args.pilot_audit_freeze))
    if args.command == "run-screen-row":
        return _summary(
            run_screen_row(
                prepared,
                ordinal=args.ordinal,
                transport_factory=_transport,
            )
        )
    if args.command == "authorize-remaining":
        return dict(authorize_remaining(prepared))
    if args.command == "run-screen-remaining":
        return _summary(
            run_screen_remaining(prepared, transport_factory=_transport)
        )
    if args.command == "freeze-screen":
        return dict(freeze_screen(prepared))
    raise ValueError("unsupported command")


def _transport(condition_id: str) -> OpenRouterTransport:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise Phase0RunnerError("OpenRouter authentication is not configured")
    policy = MAP_POLICY if condition_id == "neutral_map" else ANSWER_POLICY
    return OpenRouterTransport(api_key, policy=policy)


def _summary(summary: Phase0RunSummary) -> dict[str, object]:
    return {
        "status": "complete",
        "phase": summary.phase,
        "expected": summary.expected,
        "passed": summary.passed,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "observed_cost_usd": summary.observed_cost_usd,
    }


def _safe_error(error: BaseException) -> str:
    if isinstance(error, CostGateError):
        return "cost gate blocked dispatch"
    if isinstance(error, PilotGateError):
        return "pilot authorization gate blocked dispatch"
    if isinstance(error, AmbiguousStartedAttemptError):
        return "ambiguous started attempt requires operator review"
    if isinstance(error, Phase0RunnerError):
        return "phase zero runner rejected the requested transition"
    if isinstance(error, (FileNotFoundError, ValueError)):
        return "invalid or incomplete phase zero artifacts"
    return "phase zero command failed"


if __name__ == "__main__":
    raise SystemExit(main())
