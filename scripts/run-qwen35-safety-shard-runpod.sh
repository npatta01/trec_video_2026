#!/usr/bin/env bash
set -euo pipefail

readonly SHARD_INDEX="${1:?shard index required}"
readonly EXPECTED_ARCHIVE_SHA256="${2:?archive sha256 required}"
readonly EXPECTED_REQUEST_COUNT="${3:?request count required}"
readonly RUN_NAME="${4:?run name required}"
readonly EXPECTED_GPU_SUBSTRING="${5:?expected GPU substring required}"
readonly RUNTIME_IMAGE="${6:?runtime image identity required}"
readonly INPUT_ROOT="/workspace/input"
readonly WORK_ROOT="/root/qwen35-safety/work-$SHARD_INDEX"
readonly OUTPUT_ROOT="/workspace/output-$SHARD_INDEX"
readonly PYTHON_BIN="$(command -v python)"
readonly SOURCE_SHA256="735811649457b96f22340fb8f6bc26e6d460ac13c1781784fbddfc2abe9ee3c8"
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
[[ -f "$INPUT_ROOT/shard-$SHARD_INDEX.tar" ]] || die "shard archive is missing"
[[ -f "$INPUT_ROOT/source-v3.tar" ]] || die "source archive is missing"
mkdir -p "$WORK_ROOT/input" "$OUTPUT_ROOT"

echo "$EXPECTED_ARCHIVE_SHA256  $INPUT_ROOT/shard-$SHARD_INDEX.tar" | sha256sum -c -
echo "$SOURCE_SHA256  $INPUT_ROOT/source-v3.tar" | sha256sum -c -
if [[ ! -f "$WORK_ROOT/.prepared" ]]; then
  tar -C "$WORK_ROOT" -xf "$INPUT_ROOT/source-v3.tar"
  tar -C "$WORK_ROOT/input" -xf "$INPUT_ROOT/shard-$SHARD_INDEX.tar"
  (cd "$WORK_ROOT/input" && sha256sum -c SHA256SUMS)
  touch "$WORK_ROOT/.prepared"
fi
mkdir -p /dstack
if [[ -L /dstack/input ]]; then
  [[ "$(readlink -f /dstack/input)" == "$(readlink -f "$WORK_ROOT/input")" ]] \
    || die "/dstack/input points at another shard"
elif [[ -e /dstack/input ]]; then
  die "/dstack/input already exists"
else
  ln -s "$WORK_ROOT/input" /dstack/input
fi

"$PYTHON_BIN" - "$WORK_ROOT/input/contract.json" "$SHARD_INDEX" "$EXPECTED_REQUEST_COUNT" "$EXPECTED_GPU_SUBSTRING" <<'PY'
import json
import sys
from pathlib import Path

import torch

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if contract.get("shard_index") != int(sys.argv[2]):
    raise SystemExit("shard contract index mismatch")
if contract.get("request_count") != int(sys.argv[3]):
    raise SystemExit("shard contract request count mismatch")
if contract.get("shard_count") != 8:
    raise SystemExit("shard contract count mismatch")
if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip:
    raise SystemExit("NVIDIA CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bfloat16 is unavailable")
name = torch.cuda.get_device_name(0)
if sys.argv[4].upper() not in name.upper():
    raise SystemExit(f"unexpected GPU: {name}")
print(json.dumps({"cuda_device_name": name, "torch": torch.__version__}, sort_keys=True))
PY

if [[ ! -f "$WORK_ROOT/.dependencies-installed" ]]; then
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ https://github.com/huggingface/transformers/archive/$TRANSFORMERS_REVISION.tar.gz" \
    "qwen-vl-utils[decord]==0.0.14" "accelerate==1.12.0"
  touch "$WORK_ROOT/.dependencies-installed"
fi

set +e
PYTHONPATH="$WORK_ROOT/src" VQA_RUNTIME_IMAGE="$RUNTIME_IMAGE" \
  "$PYTHON_BIN" -m trec_video_2026.vqa_safety_runner \
    --requests "$WORK_ROOT/input/requests.json" \
    --output "$OUTPUT_ROOT/predictions.json" \
    --run-id "$RUN_NAME"
workload_rc=$?
set -e

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$workload_rc" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
(root / "terminal-status.json").write_text(
    json.dumps({"schema_version": 1, "workload_returncode": int(sys.argv[2])}, sort_keys=True) + "\n",
    encoding="utf-8",
)
paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
    for path in paths:
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
PY

exit "$workload_rc"
