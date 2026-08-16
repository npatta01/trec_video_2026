#!/usr/bin/env python3
"""CLI for the immutable Qwen empty-output diagnostic host harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trec_video_2026 import vqa_qwen_empty_output_diagnostic as diagnostic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=diagnostic.PRIVATE_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--runner-path", type=Path, default=diagnostic.DIAGNOSTIC_RUNNER_PATH)
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--data-root", type=Path, default=diagnostic.CANONICAL_DATA_ROOT)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("prepare")
    command = subparsers.add_parser("command")
    command.add_argument("--arm", required=True, choices=diagnostic.ARM_ORDER)
    record = subparsers.add_parser("record")
    record.add_argument("--arm", required=True, choices=diagnostic.ARM_ORDER)
    subparsers.add_parser("freeze")
    subparsers.add_parser("status")
    return parser


def _emit(payload: dict[str, Any]) -> None:
    """Emit aggregate-only JSON; row identities and model text never enter CLI output."""

    safe: dict[str, Any] = {"status": payload.get("status")}
    for key in (
        "decision",
        "baseline_reproducible",
        "advancing_arm",
        "attempted_arm_count",
        "required_arm_count",
        "row_count",
        "arm_count",
        "paid_compute_usd",
        "aggregate",
    ):
        if key in payload:
            safe[key] = payload[key]
    print(json.dumps(safe, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.subcommand == "prepare":
        if args.source_root is None:
            raise SystemExit("prepare requires --source-root")
        _emit(
            diagnostic.prepare_diagnostic(
                args.root, args.source_root, runner_path=args.runner_path
            )
        )
    elif args.subcommand == "command":
        if args.hub_cache is None:
            raise SystemExit("command requires --hub-cache")
        command = diagnostic.build_arm_command(
            args.root,
            args.arm,
            hub_cache=args.hub_cache,
            data_root=args.data_root,
        )
        print(json.dumps(command, ensure_ascii=False, separators=(",", ":")))
    elif args.subcommand == "record":
        if args.hub_cache is None:
            raise SystemExit("record requires --hub-cache")
        command = diagnostic.build_arm_command(
            args.root,
            args.arm,
            hub_cache=args.hub_cache,
            data_root=args.data_root,
        )
        _emit(diagnostic.record_arm(args.root, args.arm, command))
    elif args.subcommand == "freeze":
        _emit(diagnostic.freeze_diagnostic(args.root))
    else:
        _emit(diagnostic.status_diagnostic(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
