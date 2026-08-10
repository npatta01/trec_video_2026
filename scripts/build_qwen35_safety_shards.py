#!/usr/bin/env python3
"""Build deterministic private archives for parallel safety inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from trec_video_2026.vqa_safety_submission import (
    audit_label_free_payload,
    partition_requests,
    sha256_path,
    write_new_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--source-frame-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requests_path = args.requests.resolve(strict=True)
    source_root = args.source_frame_root.resolve(strict=True)
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    if not isinstance(requests, list):
        raise TypeError("requests must be a JSON list")
    audit_label_free_payload(requests)
    shards = partition_requests(requests, args.shard_count)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, shard in enumerate(shards):
        name = f"shard-{index:02d}"
        stage = output_root / name
        archive = output_root / f"{name}.tar"
        if stage.exists() or archive.exists():
            raise FileExistsError(f"refusing to overwrite {name}")
        frame_root = stage / "frames"
        rewritten = []
        linked: dict[Path, Path] = {}
        for request in shard:
            frame_paths = []
            for raw_path in request["frame_paths"]:
                source = Path(raw_path).resolve(strict=True)
                if not source.is_relative_to(source_root):
                    raise ValueError(f"frame escapes source root: {source}")
                relative = source.relative_to(source_root)
                destination = frame_root / relative
                if source not in linked:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, destination)
                    linked[source] = destination
                frame_paths.append(f"/dstack/input/frames/{relative.as_posix()}")
            rewritten.append({**request, "frame_paths": frame_paths})
        audit_label_free_payload(rewritten)
        write_new_json(stage / "requests.json", rewritten)
        contract = {
            "schema_version": 1,
            "shard_index": index,
            "shard_count": len(shards),
            "request_count": len(rewritten),
            "first_query_id": rewritten[0]["query_id"],
            "last_query_id": rewritten[-1]["query_id"],
            "source_requests_sha256": sha256_path(requests_path),
            "unique_frame_count": len(linked),
        }
        write_new_json(stage / "contract.json", contract)
        checksum_paths = sorted(path for path in stage.rglob("*") if path.is_file())
        with (stage / "SHA256SUMS").open("x", encoding="utf-8") as handle:
            for path in checksum_paths:
                handle.write(f"{sha256_path(path)}  {path.relative_to(stage)}\n")
        subprocess.run(
            [
                "tar",
                "--sort=name",
                "--mtime=@0",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "-C",
                str(stage),
                "-cf",
                str(archive.resolve()),
                ".",
            ],
            check=True,
        )
        records.append(
            {
                **contract,
                "archive": archive.name,
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": sha256_path(archive),
            }
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)
    write_new_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "shard_count": len(shards),
            "request_count": len(requests),
            "source_requests_sha256": sha256_path(requests_path),
            "shards": records,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
