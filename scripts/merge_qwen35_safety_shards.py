#!/usr/bin/env python3
"""Strictly merge ordered private Qwen safety prediction shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trec_video_2026.vqa_safety_submission import (
    merge_prediction_shards,
    sha256_path,
    write_new_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("predictions", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requests_path = args.requests.resolve(strict=True)
    prediction_paths = [path.resolve(strict=True) for path in args.predictions]
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in prediction_paths
    ]
    if not isinstance(requests, list):
        raise TypeError("requests must be a JSON list")
    if any(not isinstance(payload, dict) for payload in payloads):
        raise TypeError("prediction shards must be JSON objects")
    merged = merge_prediction_shards(requests, payloads, run_id=args.run_id)
    merged["source_requests_sha256"] = sha256_path(requests_path)
    merged["shard_artifacts"] = [
        {
            "name": path.name,
            "sha256": sha256_path(path),
            "run_id": payload["run_id"],
            "prediction_count": len(payload["predictions"]),
        }
        for path, payload in zip(prediction_paths, payloads, strict=True)
    ]
    write_new_json(args.output, merged)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prediction_count": len(merged["predictions"]),
                "sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
