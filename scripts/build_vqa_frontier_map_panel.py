#!/usr/bin/env python3
"""Build and freeze the label-free frontier video-map panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trec_video_2026.vqa_frontier_map_panel import (
    build_candidate_sheet,
    validate_panel_manifest,
    write_json_atomically,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--data-root", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--data-root", type=Path, required=True)
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "candidates":
        write_json_atomically(args.output, build_candidate_sheet(args.data_root))
        return 0

    if args.output.is_symlink():
        raise ValueError(f"refusing symlinked output: {args.output}")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    manifest = validate_panel_manifest(selection, args.data_root)
    write_json_atomically(args.output, manifest.to_payload())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
