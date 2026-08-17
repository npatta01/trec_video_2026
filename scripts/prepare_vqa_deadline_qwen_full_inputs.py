#!/usr/bin/env python3
"""Freeze the label-free full-test input contract for Qwen3.6 and Whisper.

The script records the canonical media files without copying them.  A later
upload step may sync the two canonical media directories into the single
``media/`` prefix named by this contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUESTS = REPO_ROOT / "data/vqa/submissions/qwen35-safety-20260810-a1/requests.json"
DEFAULT_WEBM_ROOT = REPO_ROOT / "data/vqa/2026/testing/videos"
DEFAULT_MKV_ROOT = REPO_ROOT / "data/vqa/2026/testing/videos_original"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/vqa/benchmarks/deadline-qwen36-full-20260816/input-freeze-a2"
)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def _resolve_media(video_id: str, webm_root: Path, mkv_root: Path) -> Path | None:
    webm_name = video_id if video_id.endswith(".webm") else f"{video_id}.webm"
    mkv_name = video_id if video_id.endswith(".mkv") else f"{video_id}.mkv"
    mp4_name = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
    candidates = (webm_root / webm_name, mkv_root / mkv_name, mkv_root / mp4_name)
    found = [path.resolve() for path in candidates if path.is_file()]
    if len(found) > 1:
        raise ValueError(f"multiple canonical media files resolve for {video_id}")
    return found[0] if found else None


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read request JSON: {path}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("request input must be a nonempty list")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ValueError("every request row must be an object")
    return payload


def prepare_inputs(
    *,
    requests_path: Path,
    webm_root: Path,
    mkv_root: Path,
    output_root: Path,
) -> dict[str, int | str]:
    rows = _load_rows(requests_path)
    query_ids = [row.get("query_id") for row in rows]
    if any(not isinstance(query_id, int) or query_id <= 0 for query_id in query_ids):
        raise ValueError("query IDs must be positive integers")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs must be unique")

    output_rows: list[dict[str, Any]] = []
    media_records: dict[str, dict[str, Any]] = {}
    query_counts: Counter[str] = Counter()
    digest_cache: dict[Path, str] = {}
    for row in rows:
        query_id = row["query_id"]
        video_id = row.get("video_id")
        question = row.get("question")
        source_sha256 = row.get("source_sha256")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"query {query_id} has no video ID")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"query {query_id} has no question")
        media_path = _resolve_media(video_id, webm_root, mkv_root)
        media_file: str | None = None
        if media_path is not None:
            if not isinstance(source_sha256, str) or len(source_sha256) != 64:
                raise ValueError(f"query {query_id} has no source SHA-256")
            digest = digest_cache.get(media_path)
            if digest is None:
                digest = sha256_file(media_path)
                digest_cache[media_path] = digest
            if digest != source_sha256:
                raise ValueError(f"media SHA-256 mismatch for {video_id}")
            media_file = f"media/{media_path.name}"
            query_counts[media_file] += 1
            existing = media_records.get(media_file)
            record = {
                "video_id": video_id,
                "path": media_file,
                "source_sha256": digest,
                "size_bytes": media_path.stat().st_size,
                "source_format": media_path.suffix.removeprefix(".").lower(),
            }
            if existing is not None and existing != record:
                raise ValueError(f"media filename collision for {media_file}")
            media_records[media_file] = record
        output_rows.append(
            {
                "query_id": query_id,
                "video_id": video_id,
                "question": question.strip(),
                "availability": "available" if media_file else "unavailable",
                "source_sha256": source_sha256,
                "media_file": media_file,
            }
        )

    media = []
    for media_file in sorted(media_records):
        record = dict(media_records[media_file])
        record["query_count"] = query_counts[media_file]
        media.append(record)
    available_count = sum(row["media_file"] is not None for row in output_rows)
    summary: dict[str, int | str] = {
        "status": "inputs_frozen",
        "query_count": len(output_rows),
        "available_query_count": available_count,
        "fallback_query_count": len(output_rows) - available_count,
        "unique_media_count": len(media),
    }
    contract = {
        "schema_version": 1,
        "label_policy": "label-free-inference-input",
        **summary,
        "media": media,
    }

    output_root.mkdir(parents=True, exist_ok=False)
    _write(output_root / "requests.json", output_rows)
    _write(output_root / "media-contract.json", contract)
    freeze = {
        "schema_version": 1,
        **summary,
        "source_requests_sha256": sha256_file(requests_path),
        "requests_sha256": sha256_file(output_root / "requests.json"),
        "media_contract_sha256": sha256_file(output_root / "media-contract.json"),
    }
    _write(output_root / "inputs-freeze.json", freeze)
    checksum_paths = [
        output_root / "inputs-freeze.json",
        output_root / "media-contract.json",
        output_root / "requests.json",
    ]
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}" for path in sorted(checksum_paths)
    ]
    (output_root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--webm-root", type=Path, default=DEFAULT_WEBM_ROOT)
    parser.add_argument("--mkv-root", type=Path, default=DEFAULT_MKV_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare_inputs(
        requests_path=args.requests,
        webm_root=args.webm_root,
        mkv_root=args.mkv_root,
        output_root=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
