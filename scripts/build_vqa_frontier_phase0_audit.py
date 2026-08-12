#!/usr/bin/env python3
"""Safe CLI for the private Frontier Phase 0 native-evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from trec_video_2026.vqa_frontier_phase0_audit import (
    build_audit_packet,
    freeze_pilot_audit,
    freeze_screen_audit,
    validate_audit_packet,
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _dispatch(args)
    except Exception:
        print(
            json.dumps(
                {"status": "error", "error": "invalid or incomplete audit artifacts"}
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_vqa_frontier_phase0_audit.py")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--prepared", type=Path, required=True)
    build.add_argument("--phase", choices=("pilot", "screen"), required=True)
    build.add_argument("--output", type=Path, required=True)

    for name in ("validate", "freeze-pilot", "freeze-screen"):
        command = commands.add_parser(name)
        command.add_argument("--audit", type=Path, required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "build":
        packet = build_audit_packet(
            args.prepared, phase=args.phase, output=args.output
        )
        manifest = json.loads(packet.packet_manifest.read_text(encoding="utf-8"))
        rows = manifest.get("rows", []) if isinstance(manifest, dict) else []
        return {
            "status": "built",
            "phase": args.phase,
            "row_count": len(rows),
            "manifest_sha256": _sha256(packet.packet_manifest),
        }
    if args.command == "validate":
        return dict(validate_audit_packet(args.audit))
    if args.command == "freeze-pilot":
        return _safe_freeze(freeze_pilot_audit(args.audit))
    if args.command == "freeze-screen":
        return _safe_freeze(freeze_screen_audit(args.audit))
    raise ValueError("unsupported command")


def _safe_freeze(payload: dict[str, object]) -> dict[str, object]:
    bindings = payload.get("bindings")
    return {
        "status": payload.get("status"),
        "phase": bindings.get("phase") if isinstance(bindings, dict) else None,
        "decision": bindings.get("decision") if isinstance(bindings, dict) else None,
        "artifact_sha256": payload.get("artifact_sha256"),
        "bindings_sha256": payload.get("bindings_sha256"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
