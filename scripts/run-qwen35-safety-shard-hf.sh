#!/usr/bin/env bash
set -euo pipefail

readonly SHARD_INDEX="${1:?shard index required}"
readonly EXPECTED_ARCHIVE_SHA256="${2:?archive sha256 required}"
readonly EXPECTED_REQUEST_COUNT="${3:?request count required}"
readonly RUN_NAME="${4:?run name required}"
readonly INPUT_MOUNT="/mnt/input"
readonly OUTPUT_MOUNT="/mnt/output"
readonly WORK_ROOT="/workspace/qwen35-safety"
readonly PYTHON_BIN="/opt/conda/bin/python"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ "$SHARD_INDEX" =~ ^0[0-7]$ ]] || die "shard index must be 00 through 07"
[[ "$EXPECTED_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid archive hash"
[[ "$EXPECTED_REQUEST_COUNT" =~ ^[0-9]+$ ]] || die "invalid request count"
[[ "$RUN_NAME" == "qwen35-safety-20260810-a1-shard-$SHARD_INDEX" ]] \
  || die "run name mismatch"
[[ -x "$PYTHON_BIN" ]] || die "pinned image interpreter is missing"
[[ -f "$INPUT_MOUNT/shard-$SHARD_INDEX.tar" ]] || die "shard archive is missing"
[[ -f "$INPUT_MOUNT/source-v2.tar" ]] || die "source archive is missing"
mkdir -p "$WORK_ROOT/input" "$WORK_ROOT/output" "$OUTPUT_MOUNT"
[[ -z "$(find "$OUTPUT_MOUNT" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  || die "output mount is not empty"

echo "$EXPECTED_ARCHIVE_SHA256  $INPUT_MOUNT/shard-$SHARD_INDEX.tar" | sha256sum -c -
echo "${EXPECTED_SOURCE_SHA256:?EXPECTED_SOURCE_SHA256 is required}  $INPUT_MOUNT/source-v2.tar" | sha256sum -c -
tar -C "$WORK_ROOT" -xf "$INPUT_MOUNT/source-v2.tar"
tar -C "$WORK_ROOT/input" -xf "$INPUT_MOUNT/shard-$SHARD_INDEX.tar"
mkdir -p /dstack
[[ ! -e /dstack/input ]] || die "/dstack/input already exists"
ln -s "$WORK_ROOT/input" /dstack/input
(cd "$WORK_ROOT/input" && sha256sum -c SHA256SUMS)

"$PYTHON_BIN" - "$WORK_ROOT/input/contract.json" "$SHARD_INDEX" "$EXPECTED_REQUEST_COUNT" <<'PY'
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

"$PYTHON_BIN" - <<'PY'
import json
import torch

if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
    raise SystemExit("NVIDIA CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bfloat16 is unavailable")
name = torch.cuda.get_device_name(0)
expected = __import__("os").environ.get("EXPECTED_GPU_SUBSTRING", "L40S").upper()
if expected not in name.upper():
    raise SystemExit(f"expected GPU containing {expected!r}, found {name}")
print(json.dumps({"cuda_device_name": name, "torch": torch.__version__}, sort_keys=True))
PY

"$PYTHON_BIN" -m pip install --no-cache-dir \
  "transformers @ https://github.com/huggingface/transformers/archive/$TRANSFORMERS_REVISION.tar.gz" \
  "qwen-vl-utils[decord]==0.0.14" "accelerate==1.12.0"

set +e
PYTHONPATH="$WORK_ROOT/src" \
VQA_RUNTIME_IMAGE="${VQA_RUNTIME_IMAGE:?VQA_RUNTIME_IMAGE is required}" \
"$PYTHON_BIN" -m trec_video_2026.vqa_safety_runner \
  --requests "$WORK_ROOT/input/requests.json" \
  --output "$WORK_ROOT/output/predictions.json" \
  --run-id "$RUN_NAME"
workload_rc=$?
set -e

"$PYTHON_BIN" - "$WORK_ROOT/output" "$workload_rc" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
(root / "terminal-status.json").write_text(
    json.dumps(
        {"schema_version": 1, "workload_returncode": int(sys.argv[2])},
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
paths = sorted(path for path in root.iterdir() if path.is_file())
with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path.name}\n")
PY

cp -a "$WORK_ROOT/output/." "$OUTPUT_MOUNT/"
sync
exit "$workload_rc"
