#!/usr/bin/env bash
set -euo pipefail

readonly SHARD_INDEX="${1:?shard index required}"
readonly EXPECTED_ARCHIVE_SHA256="${2:?archive sha256 required}"
readonly EXPECTED_REQUEST_COUNT="${3:?request count required}"
readonly EXPECTED_RUN_NAME="${4:?run name required}"
readonly INPUT_ROOT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/artifacts/qwen35-safety-20260810-a1-shards-v1"
readonly OUTPUT_ROOT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments"
readonly OUTPUT_URI="$OUTPUT_ROOT_URI/$EXPECTED_RUN_NAME"
readonly PYTHON_BIN="/opt/conda/bin/python"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly INPUT_DIR="/dstack/input"
readonly OUTPUT_DIR="/dstack/output"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ "$SHARD_INDEX" =~ ^0[0-7]$ ]] || die "shard index must be 00 through 07"
[[ "$EXPECTED_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid archive hash"
[[ "$EXPECTED_REQUEST_COUNT" =~ ^[0-9]+$ ]] || die "invalid request count"
[[ "$EXPECTED_RUN_NAME" == "qwen35-safety-20260810-a1-shard-$SHARD_INDEX" ]] \
  || die "run name mismatch"
[[ "${VQA_RUNTIME_IMAGE:-}" == "$IMAGE" ]] || die "runtime image mismatch"
[[ -x "$PYTHON_BIN" ]] || die "pinned Python interpreter is missing"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
"$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: \
  "uv==0.11.3" "huggingface-hub==1.26.0"
"$PYTHON_BIN" -m pip install --no-cache-dir \
  "transformers @ https://github.com/huggingface/transformers/archive/$TRANSFORMERS_REVISION.tar.gz" \
  "qwen-vl-utils[decord]==0.0.14" "accelerate==1.12.0"

readonly HF_BIN="/opt/conda/bin/hf"
[[ -x "$HF_BIN" ]] || die "pinned Hugging Face CLI is missing"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}" "$HF_BIN" auth whoami >/dev/null
listing="$(HF_TOKEN="$HF_TOKEN" "$HF_BIN" buckets list "$OUTPUT_URI" --format quiet)"
[[ -z "$listing" ]] || die "remote output prefix already exists: $OUTPUT_URI"

archive="/dstack/shard-$SHARD_INDEX.tar"
HF_TOKEN="$HF_TOKEN" "$HF_BIN" buckets cp \
  "$INPUT_ROOT_URI/shard-$SHARD_INDEX.tar" "$archive"
echo "$EXPECTED_ARCHIVE_SHA256  $archive" | sha256sum -c -
tar -C "$INPUT_DIR" -xf "$archive"
(cd "$INPUT_DIR" && sha256sum -c SHA256SUMS)

"$PYTHON_BIN" - "$INPUT_DIR/contract.json" "$SHARD_INDEX" "$EXPECTED_REQUEST_COUNT" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if contract.get("shard_index") != int(sys.argv[2]):
    raise SystemExit("shard contract index mismatch")
if contract.get("request_count") != int(sys.argv[3]):
    raise SystemExit("shard contract request count mismatch")
if contract.get("shard_count") != 8:
    raise SystemExit("shard contract count mismatch")
PY

set +e
VQA_RUNTIME_IMAGE="$IMAGE" "$PYTHON_BIN" -m trec_video_2026.vqa_safety_runner \
  --requests "$INPUT_DIR/requests.json" \
  --output "$OUTPUT_DIR/predictions.json" \
  --run-id "$EXPECTED_RUN_NAME"
workload_rc=$?
set -e

"$PYTHON_BIN" - "$OUTPUT_DIR" "$workload_rc" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = {"schema_version": 1, "workload_returncode": int(sys.argv[2])}
(root / "terminal-status.json").write_text(json.dumps(status, sort_keys=True) + "\n")
paths = sorted(path for path in root.iterdir() if path.is_file())
with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path.name}\n")
PY

set +e
HF_TOKEN="$HF_TOKEN" "$HF_BIN" buckets sync "$OUTPUT_DIR" "$OUTPUT_URI"
sync_rc=$?
set -e
(( workload_rc == 0 )) || exit "$workload_rc"
exit "$sync_rc"
