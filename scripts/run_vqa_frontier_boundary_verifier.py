#!/usr/bin/env python3
"""CLI for the one-call question-blind frontier boundary verifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from trec_video_2026.vqa_frontier_boundary_verifier import (
    BoundaryVerifierError,
    VERIFIER_POLICY,
    freeze_verifier,
    load_verifier,
    preflight_verifier,
    prepare_verifier,
    run_verifier,
    status_verifier,
)
from trec_video_2026.vqa_submission_openrouter import OpenRouterTransport


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except Exception as error:
        print(
            json.dumps({"status": "error", "error": _safe_error(error)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_frontier_boundary_verifier.py")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--prepared", type=Path, required=True)
    prepare.add_argument("--pilot-audit-freeze", type=Path, required=True)
    prepare.add_argument("--video-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for command_name in ("preflight", "run", "freeze", "status"):
        command = commands.add_parser(command_name)
        command.add_argument("--prepared", type=Path, required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "prepare":
        prepared = prepare_verifier(
            args.prepared,
            args.pilot_audit_freeze,
            args.video_id,
            args.output,
        )
        return {
            "status": "prepared",
            "video_id": prepared.manifest["video_id"],
            "source_sha256": prepared.manifest["source_sha256"],
            "draft_map_sha256": prepared.draft_map.sha256,
        }
    prepared = load_verifier(args.prepared)
    if args.command == "preflight":
        return preflight_verifier(prepared)
    if args.command == "status":
        return status_verifier(prepared)
    if args.command == "freeze":
        return freeze_verifier(prepared)
    if args.command == "run":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise BoundaryVerifierError("OpenRouter authentication is not configured")
        transport = OpenRouterTransport(api_key, policy=VERIFIER_POLICY)
        return run_verifier(prepared, transport)
    raise ValueError("unsupported verifier command")


def _safe_error(error: BaseException) -> str:
    if isinstance(error, BoundaryVerifierError):
        return "boundary verifier rejected the requested transition"
    if isinstance(error, FileExistsError):
        return "boundary verifier artifact already exists"
    if isinstance(error, (FileNotFoundError, TypeError, ValueError)):
        return "invalid or incomplete boundary verifier artifacts"
    return "boundary verifier command failed"


if __name__ == "__main__":
    raise SystemExit(main())
