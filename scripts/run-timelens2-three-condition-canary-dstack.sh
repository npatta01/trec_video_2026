#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUN_ID="timelens2-three-condition-canary-20260801-a3"
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$EXPECTED_RUN_ID"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_SELECTOR_ID="MCG-NJU/TimeLens2-2B"
readonly EXPECTED_SELECTOR_REVISION="90dd30b3eeff3520a30c830eff2209f4ad788ca9"
readonly EXPECTED_ANSWER_ID="Qwen/Qwen3.5-4B"
readonly EXPECTED_ANSWER_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_VERSION="5.15.0.dev0"
readonly TRANSFORMERS_SOURCE_URL="https://github.com/huggingface/transformers/archive/cbf4d720ec734edb77d25452b2790c7e4be2f8d7.tar.gz"
readonly QWEN_VL_UTILS_VERSION="0.0.14"
readonly PYAV_VERSION="16.0.1"
readonly ACCELERATE_VERSION="1.12.0"
readonly IMAGEIO_FFMPEG_VERSION="0.6.0"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.26.0"

HF_ACCESS_TOKEN="${HF_TOKEN:-}"
readonly HF_ACCESS_TOKEN
unset HF_TOKEN
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
OUTPUT_DIR="/dstack/output"
SELECTOR_MODEL_DIR="/dstack/models/timelens2"
ANSWER_MODEL_DIR="/dstack/models/qwen35"

die() { echo "error: $*" >&2; exit 2; }

validate_environment() {
  [[ -n "$HF_ACCESS_TOKEN" ]] || die "HF_TOKEN is unavailable"
  [[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || die "invalid image interpreter"
  [[ "${EXPERIMENT_RUN_ID:-}" == "$EXPECTED_RUN_ID" ]] || die "run ID mismatch"
  [[ "${HF_REMOTE_URI:-}" == "$EXPECTED_REMOTE_URI" ]] || die "remote URI mismatch"
  [[ "${RUNTIME_IMAGE:-}" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image mismatch"
  [[ "${SELECTOR_MODEL_ID:-}" == "$EXPECTED_SELECTOR_ID" ]] || die "selector model mismatch"
  [[ "${SELECTOR_MODEL_REVISION:-}" == "$EXPECTED_SELECTOR_REVISION" ]] || die "selector revision mismatch"
  [[ "${ANSWER_MODEL_ID:-}" == "$EXPECTED_ANSWER_ID" ]] || die "answer model mismatch"
  [[ "${ANSWER_MODEL_REVISION:-}" == "$EXPECTED_ANSWER_REVISION" ]] || die "answer revision mismatch"
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "uv==$UV_VERSION"
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ $TRANSFORMERS_SOURCE_URL" "qwen-vl-utils[decord]==$QWEN_VL_UTILS_VERSION" \
    "av==$PYAV_VERSION" "accelerate==$ACCELERATE_VERSION" \
    "imageio-ffmpeg==$IMAGEIO_FFMPEG_VERSION"
  verify_transformers_runtime "$OUTPUT_DIR/runtime-capability.json"
}

verify_transformers_runtime() {
  local record_path="${1:-}"
  "$PYTHON_BIN" - "$TRANSFORMERS_VERSION" "$TRANSFORMERS_REVISION" "$record_path" <<'PY'
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

expected_version, revision, record_path = sys.argv[1:4]
if transformers.__version__ != expected_version:
    raise SystemExit(
        "qwen3_5 runtime capability check failed: "
        f"expected Transformers {expected_version}, got {transformers.__version__}"
    )
try:
    config = AutoConfig.for_model("qwen3_5")
except Exception as error:
    raise SystemExit(f"qwen3_5 runtime capability check failed: {error}") from error
if getattr(config, "model_type", None) != "qwen3_5":
    raise SystemExit("qwen3_5 runtime capability check returned wrong model type")
if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip is not None:
    raise SystemExit("qwen3_5 runtime requires NVIDIA CUDA")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("qwen3_5 runtime requires CUDA bfloat16 support")
if AutoProcessor is None or AutoModelForImageTextToText is None:
    raise SystemExit("qwen3_5 runtime auto classes are unavailable")
record = {
    "cuda_available": True,
    "cuda_bf16_supported": True,
    "qwen3_5_model_type": config.model_type,
    "transformers_revision": revision,
    "transformers_version": transformers.__version__,
}
payload = json.dumps(record, sort_keys=True) + "\n"
if record_path:
    Path(record_path).write_text(payload, encoding="utf-8")
print(payload, end="")
PY
}

hf_cli() { HF_TOKEN="$HF_ACCESS_TOKEN" uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"; }

finalize() {
  local workload_rc="$1"
  mkdir -p "$OUTPUT_DIR"
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"workload_returncode": int(sys.argv[2])}, sort_keys=True) + "\n")
PY
  (cd "$OUTPUT_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$EXPECTED_REMOTE_URI/output"
  local sync_rc=$?
  set -e
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

run_workload() {
  validate_environment
  bootstrap_runtime
  hf_cli version
  hf_cli auth whoami
  local listing
  listing="$(hf_cli buckets list "$EXPECTED_REMOTE_URI/output" --format quiet)" || die "could not list output prefix"
  [[ -z "$listing" ]] || die "immutable output prefix already exists"
  mkdir -p "$SELECTOR_MODEL_DIR" "$ANSWER_MODEL_DIR" "$OUTPUT_DIR"
  hf_cli download "$EXPECTED_SELECTOR_ID" --revision "$EXPECTED_SELECTOR_REVISION" --local-dir "$SELECTOR_MODEL_DIR"
  hf_cli download "$EXPECTED_ANSWER_ID" --revision "$EXPECTED_ANSWER_REVISION" --local-dir "$ANSWER_MODEL_DIR"
  local ffmpeg
  ffmpeg="$("$PYTHON_BIN" -c 'from imageio_ffmpeg import get_ffmpeg_exe; print(get_ffmpeg_exe())')"
  exec "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.timelens2_three_condition canary \
    --selector-model-dir "$SELECTOR_MODEL_DIR" --answer-model-dir "$ANSWER_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" --ffmpeg "$ffmpeg"
}

case "${1:-}" in
  validate) validate_environment ;;
  verify-runtime) verify_transformers_runtime ;;
  run)
    mkdir -p "$OUTPUT_DIR"
    set +e
    (set -euo pipefail; run_workload)
    workload_rc=$?
    set -e
    finalize "$workload_rc"
    ;;
  *) die "usage: $0 validate|verify-runtime|run" ;;
esac
