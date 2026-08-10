#!/usr/bin/env python3
"""Materialize shared uniform-12 evidence and label-free safety requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trec_video_2026.vqa_safety_submission import (
    extract_uniform_frames,
    prepare_inference_requests,
    write_new_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        manifest = {**manifest, "queries": manifest["queries"][: args.limit]}
    requests = prepare_inference_requests(
        manifest,
        dataset_root=args.dataset_root,
        frame_root=args.frame_root,
        extract_frames=extract_uniform_frames,
        workers=args.workers,
    )
    write_new_json(args.output, requests)
    print(
        json.dumps(
            {
                "request_count": len(requests),
                "available_count": sum(
                    row["availability"] == "available" for row in requests
                ),
                "unavailable_count": sum(
                    row["availability"] == "unavailable" for row in requests
                ),
                "unique_evidence_sets": len(
                    {row["video_id"] for row in requests if row["frame_paths"]}
                ),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
