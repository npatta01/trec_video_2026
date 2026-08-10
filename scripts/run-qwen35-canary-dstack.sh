#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUN_NAME="qwen35-4b-cuda-canary-20260803-c2"
readonly EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-frozen30-20260801-a1/inputs"
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$EXPECTED_RUN_NAME"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly MODEL_ID="Qwen/Qwen3.5-4B"
readonly MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_VERSION="5.15.0.dev0"
readonly TRANSFORMERS_SOURCE_URL="https://github.com/huggingface/transformers/archive/$TRANSFORMERS_REVISION.tar.gz"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.26.0"
readonly PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
readonly INPUT_DIR="${INPUT_DIR:-/dstack/input}"
readonly FRAME_DIR="${FRAME_DIR:-/dstack/frames}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
readonly MANIFEST="${MANIFEST:-/dstack/run/configs/qwen35_canary_3_label_free.json}"

die() {
  echo "error: $*" >&2
  exit 2
}

hf_cli() {
  HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}" \
    uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

verify_environment() {
  [[ -x "$PYTHON_BIN" ]] || die "missing image interpreter: $PYTHON_BIN"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image interpreter identity mismatch"
  [[ "${EXPERIMENT_RUN_NAME:-}" == "$EXPECTED_RUN_NAME" ]] \
    || die "experiment name mismatch"
  [[ "${HF_INPUT_URI:-}" == "$EXPECTED_INPUT_URI" ]] \
    || die "input URI mismatch"
  [[ "${HF_REMOTE_URI:-}" == "$EXPECTED_REMOTE_URI" ]] \
    || die "remote URI mismatch"
  [[ "${VQA_RUNTIME_IMAGE:-}" == "$EXPECTED_RUNTIME_IMAGE" ]] \
    || die "runtime image mismatch"
  [[ -f "$MANIFEST" ]] || die "label-free manifest is missing: $MANIFEST"
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "uv==$UV_VERSION"
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ $TRANSFORMERS_SOURCE_URL" \
    "qwen-vl-utils[decord]==0.0.14" \
    "av==16.0.1" \
    "accelerate==1.12.0" \
    "imageio-ffmpeg==0.6.0"
  "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" <<'PY'
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig

if transformers.__version__ != "5.15.0.dev0":
    raise SystemExit(f"unexpected transformers version: {transformers.__version__}")
if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
    raise SystemExit("NVIDIA CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bfloat16 is unavailable")
config = AutoConfig.for_model("qwen3_5")
record = {
    "model_type": config.model_type,
    "transformers_version": transformers.__version__,
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_bf16_supported": True,
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY
}

remote_prefix_is_empty() {
  local listing list_rc
  set +e
  listing="$(hf_cli buckets list "$EXPECTED_REMOTE_URI" --format quiet 2>"$OUTPUT_DIR/remote-list.stderr")"
  list_rc=$?
  set -e
  if (( list_rc != 0 )); then
    sed -n '1,120p' "$OUTPUT_DIR/remote-list.stderr" >&2
    die "could not inspect remote output prefix (hf rc=$list_rc)"
  fi
  [[ -z "$listing" ]] || die "remote output prefix already exists: $EXPECTED_REMOTE_URI"
}

prepare_input() {
  mkdir -p "$INPUT_DIR" "$FRAME_DIR" "$OUTPUT_DIR"
  hf_cli buckets sync "$EXPECTED_INPUT_URI" "$INPUT_DIR"
  [[ -f "$INPUT_DIR/contract.json" ]] || die "input contract is missing"
  [[ -f "$INPUT_DIR/INPUTS_COMPLETE.json" ]] || die "input completion marker is missing"
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_prepare_requests.py \
    --manifest "$MANIFEST" \
    --input-root "$INPUT_DIR" \
    --frame-root "$FRAME_DIR" \
    --output "$OUTPUT_DIR/requests.json" \
    --run-id "$EXPECTED_RUN_NAME" \
    --condition-id qwen35-4b-12f-canary \
    --count 3 \
    --frame-count 12
}

run_workload() {
  local started_ns child_rc
  started_ns="$(date +%s%N)"
  set +e
  "$PYTHON_BIN" -m trec_video_2026.vqa_transformers_runner \
    --model-id "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --model-class image-text \
    --max-new-tokens 64 \
    < "$OUTPUT_DIR/requests.json" \
    > "$OUTPUT_DIR/predictions.json" \
    2> "$OUTPUT_DIR/inference.stderr"
  child_rc=$?
  set -e
  local finished_ns=$(( $(date +%s%N) ))
  "$PYTHON_BIN" - "$OUTPUT_DIR/summary.json" "$child_rc" "$started_ns" "$finished_ns" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
child_rc = int(sys.argv[2])
started_ns = int(sys.argv[3])
finished_ns = int(sys.argv[4])
payload = {
    "workload_returncode": child_rc,
    "wall_seconds": (finished_ns - started_ns) / 1_000_000_000,
}
predictions = Path("/dstack/output/predictions.json")
if predictions.is_file():
    try:
        rows = json.loads(predictions.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        rows = None
    if isinstance(rows, list):
        payload["rows"] = len(rows)
        payload["statuses"] = {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status") for row in rows})
        }
        elapsed = [row["elapsed_seconds"] for row in rows if isinstance(row.get("elapsed_seconds"), (int, float))]
        if elapsed:
            payload["row_elapsed_seconds"] = {
                "sum": sum(elapsed),
                "mean": sum(elapsed) / len(elapsed),
                "min": min(elapsed),
                "max": max(elapsed),
            }
        devices = sorted({row.get("device") for row in rows})
        payload["devices"] = devices
        payload["model_load_seconds"] = sorted({row.get("model_load_seconds") for row in rows})
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  return "$child_rc"
}

finalize() {
  local workload_rc="$1" sync_rc
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"workload_returncode": int(sys.argv[2])}, sort_keys=True) + "\n")
PY
  (cd "$OUTPUT_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$EXPECTED_REMOTE_URI/output"
  sync_rc=$?
  set -e
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" "$sync_rc" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"workload_returncode": int(sys.argv[2]), "sync_returncode": int(sys.argv[3])}, sort_keys=True) + "\n")
PY
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

main() {
  mkdir -p "$OUTPUT_DIR"
  verify_environment
  bootstrap_runtime
  uvx hf version > "$OUTPUT_DIR/hf-version.txt"
  uvx hf auth whoami > "$OUTPUT_DIR/hf-whoami.txt"
  remote_prefix_is_empty
  prepare_input
  "$PYTHON_BIN" -m trec_video_2026.vqa_transformers_runner --help > "$OUTPUT_DIR/child-help.txt"
  set +e
  run_workload
  local workload_rc=$?
  set -e
  finalize "$workload_rc"
}

main "$@"
