#!/usr/bin/env python3
"""Prepare the private label-free TRECVID 2026 Qwen safety manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trec_video_2026.vqa_safety_submission import build_test_manifest, write_new_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_test_manifest(args.topics, args.dataset_root)
    write_new_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "query_count": manifest["query_count"],
                "video_count": manifest["video_count"],
                "available_video_count": manifest["available_video_count"],
                "unavailable_video_count": manifest["unavailable_video_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
