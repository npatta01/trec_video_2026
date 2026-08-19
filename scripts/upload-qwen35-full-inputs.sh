#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly RUN_ID="qwen35-4b-12f-full-20260803-b1"
readonly DATA_ROOT="/home/npatta01/data/competitions/trec_video_2026/data"
readonly MEDIA_ROOT="$DATA_ROOT/vqa/2025/training/videos_original"
readonly MANIFEST="configs/vqa/qwen35_vqa2025_475_label_free.json"
readonly REMOTE_URI="hf://buckets/Npatta01/trec_video_2026/experiments/$RUN_ID-inputs"
readonly HF_CLI_VERSION="1.26.0"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ -L data ]] || die "data must be a symlink"
[[ "$(readlink -f data)" == "$EXPECTED_DATA_TARGET" ]] \
  || die "data symlink target mismatch"
[[ -d "$MEDIA_ROOT" ]] || die "missing VQA training media root"
[[ -f "$MANIFEST" ]] || die "missing label-free manifest"

hf_cli() {
  uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

existing="$(hf_cli buckets list "$REMOTE_URI" --format quiet)"
[[ -z "$existing" ]] || die "immutable input prefix already exists: $REMOTE_URI"

temp_dir="$(mktemp -d /tmp/qwen35-full-inputs.XXXXXX)"
contract="$temp_dir/contract.json"
marker="$temp_dir/INPUTS_COMPLETE.json"

python3 - "$MANIFEST" "$MEDIA_ROOT" "$contract" "$marker" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
media_root = Path(sys.argv[2]).resolve(strict=True)
contract_path = Path(sys.argv[3])
marker_path = Path(sys.argv[4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
examples = manifest.get("examples")
if manifest.get("label_policy") != "label-free-inference-input":
    raise SystemExit("manifest is not label-free")
if not isinstance(examples, list) or len(examples) != 475:
    raise SystemExit("manifest must contain exactly 475 examples")

objects: list[dict[str, object]] = []
seen: set[str] = set()
for example in examples:
    video_id = example.get("video_id")
    if not isinstance(video_id, str) or video_id in seen:
        raise SystemExit("manifest IDs must be unique strings")
    seen.add(video_id)
    matches = sorted(media_root.glob(video_id + ".*"))
    matches = [
        path
        for path in matches
        if path.suffix.lower() in {".mkv", ".mp4", ".webm", ".mov"}
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one media file for {video_id}: {matches}")
    path = matches[0]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    relative = path.relative_to(media_root.parent.parent.parent.parent)
    objects.append(
        {
            "video_id": video_id,
            "path": relative.as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    )

objects.sort(key=lambda item: str(item["video_id"]))
if len(objects) != 475 or len({item["video_id"] for item in objects}) != 475:
    raise SystemExit("media contract must contain exactly 475 unique objects")
contract = {
    "schema_version": 1,
    "run_id": "qwen35-4b-12f-full-20260803-b1",
    "media_count": len(objects),
    "media_root": "vqa/2025/training/videos_original",
    "objects": objects,
}
contract_bytes = (
    json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
).encode()
contract_path.write_bytes(contract_bytes)
marker = {
    "schema_version": 1,
    "input_count": len(objects),
    "object_count": len(objects) + 2,
    "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
    "media_count": len(objects),
    "media_bytes": sum(int(item["size_bytes"]) for item in objects),
}
marker_path.write_text(
    json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {"media_count": len(objects), "media_bytes": marker["media_bytes"]},
        sort_keys=True,
    )
)
PY

printf '%s\n' 'Uploading contract...'
hf_cli buckets cp "$contract" "$REMOTE_URI/inputs/contract.json"
printf '%s\n' 'Uploading 475 media files via bucket sync...'
hf_cli buckets sync "$MEDIA_ROOT" "$REMOTE_URI/inputs/vqa/2025/training/videos_original" \
  --include '*.mkv' \
  --include '*.mp4' \
  --include '*.webm' \
  --include '*.mov'
printf '%s\n' 'Uploading completion marker last...'
hf_cli buckets cp "$marker" "$REMOTE_URI/inputs/INPUTS_COMPLETE.json"

remote_count="$(hf_cli buckets list "$REMOTE_URI/inputs" --recursive --format quiet | wc -l)"
[[ "$remote_count" -eq 477 ]] \
  || die "expected 477 remote input objects, found $remote_count"

printf '%s\n' "Input bundle ready: $REMOTE_URI (objects=$remote_count)"
