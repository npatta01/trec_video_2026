#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_RUN_ID="timelens2-selector-quality-20260801-a2"
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/timelens2-selector-quality-20260801-a2"
readonly EXPECTED_SELECTOR_ID="MCG-NJU/TimeLens2-2B"
readonly EXPECTED_SELECTOR_REVISION="90dd30b3eeff3520a30c830eff2209f4ad788ca9"
readonly EXPECTED_ANSWER_ID="Qwen/Qwen3.5-4B"
readonly EXPECTED_ANSWER_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
readonly EXPECTED_SOURCE_SHA256="d2cfec431652bcf7c6c83be59a4932fc0ed75016d6a3dc973e49cebdd9ed0241"
readonly TRANSFORMERS_VERSION="4.57.6"
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
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
SELECTOR_MODEL_ID="${SELECTOR_MODEL_ID:-}"
SELECTOR_MODEL_REVISION="${SELECTOR_MODEL_REVISION:-}"
ANSWER_MODEL_ID="${ANSWER_MODEL_ID:-}"
ANSWER_MODEL_REVISION="${ANSWER_MODEL_REVISION:-}"
SOURCE_SHA256="${SOURCE_SHA256:-}"

readonly INPUT_DIR="/dstack/input"
readonly OUTPUT_DIR="/dstack/output"
readonly SELECTOR_MODEL_DIR="/dstack/models/timelens2"
readonly ANSWER_MODEL_DIR="/dstack/models/qwen35"
HF_INPUT_URI="${HF_REMOTE_URI}/input"
HF_OUTPUT_URI="${HF_REMOTE_URI}/output"

die() {
  echo "error: $*" >&2
  exit 2
}

validate_environment() {
  [[ -n "$HF_ACCESS_TOKEN" ]] || die "HF_TOKEN is unavailable"
  [[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || die "invalid image interpreter"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image interpreter identity mismatch"
  [[ "$EXPERIMENT_RUN_ID" == "$EXPECTED_RUN_ID" ]] || die "run ID mismatch"
  [[ "$HF_REMOTE_URI" == "$EXPECTED_REMOTE_URI" ]] || die "remote URI mismatch"
  [[ "$RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image mismatch"
  [[ "$SELECTOR_MODEL_ID" == "$EXPECTED_SELECTOR_ID" ]] || die "selector model ID mismatch"
  [[ "$SELECTOR_MODEL_REVISION" == "$EXPECTED_SELECTOR_REVISION" ]] \
    || die "selector model revision mismatch"
  [[ "$ANSWER_MODEL_ID" == "$EXPECTED_ANSWER_ID" ]] || die "answer model ID mismatch"
  [[ "$ANSWER_MODEL_REVISION" == "$EXPECTED_ANSWER_REVISION" ]] \
    || die "answer model revision mismatch"
  [[ "$SOURCE_SHA256" == "$EXPECTED_SOURCE_SHA256" ]] || die "source SHA-256 mismatch"
  [[ "$HF_INPUT_URI" == "$EXPECTED_REMOTE_URI/input" ]] || die "input URI mismatch"
  [[ "$HF_OUTPUT_URI" == "$EXPECTED_REMOTE_URI/output" ]] || die "output URI mismatch"
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "uv==$UV_VERSION"
  command -v uvx >/dev/null || die "uvx is unavailable"
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: \
    "transformers==$TRANSFORMERS_VERSION" \
    "qwen-vl-utils[decord]==$QWEN_VL_UTILS_VERSION" \
    "av==$PYAV_VERSION" \
    "accelerate==$ACCELERATE_VERSION" \
    "imageio-ffmpeg==$IMAGEIO_FFMPEG_VERSION"
  "$PYTHON_BIN" - "$TRANSFORMERS_VERSION" "$QWEN_VL_UTILS_VERSION" \
    "$PYAV_VERSION" "$IMAGEIO_FFMPEG_VERSION" <<'PY'
import importlib.metadata
import sys
import torch

assert importlib.metadata.version("transformers") == sys.argv[1]
assert importlib.metadata.version("qwen-vl-utils") == sys.argv[2]
assert importlib.metadata.version("av") == sys.argv[3]
assert importlib.metadata.version("imageio-ffmpeg") == sys.argv[4]
assert torch.cuda.is_available()
assert torch.version.cuda
assert torch.version.hip is None
assert torch.cuda.is_bf16_supported()
PY
}

hf_cli() {
  HF_TOKEN="$HF_ACCESS_TOKEN" uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

require_remote_input_and_empty_output() {
  local input_listing output_listing
  input_listing="$(hf_cli buckets list "$HF_INPUT_URI" --format quiet)" \
    || die "could not list immutable input prefix"
  [[ -n "$input_listing" ]] || die "immutable input prefix is empty"
  output_listing="$(hf_cli buckets list "$HF_OUTPUT_URI" --format quiet)" \
    || die "could not list output prefix"
  [[ -z "$output_listing" ]] || die "immutable output prefix already exists"
}

download_and_verify_input() {
  mkdir -p "$INPUT_DIR"
  hf_cli buckets sync "$HF_INPUT_URI" "$INPUT_DIR"
  [[ -f "$INPUT_DIR/SHA256SUMS" ]] || die "input SHA256SUMS is missing"
  if find "$INPUT_DIR" -type l -print -quit | grep -q .; then
    die "input bundle contains a symlink"
  fi
  (
    cd "$INPUT_DIR"
    sha256sum -c SHA256SUMS
  )
  [[ -f "$INPUT_DIR/source/inference-manifest.json" ]] \
    || die "frozen label-free source is missing"
  [[ "$(sha256sum "$INPUT_DIR/source/inference-manifest.json" | cut -d' ' -f1)" \
      == "$EXPECTED_SOURCE_SHA256" ]] || die "frozen source hash mismatch"
  chmod -R a-w "$INPUT_DIR"
}

download_models() {
  mkdir -p "$SELECTOR_MODEL_DIR" "$ANSWER_MODEL_DIR"
  hf_cli download "$SELECTOR_MODEL_ID" --revision "$SELECTOR_MODEL_REVISION" \
    --local-dir "$SELECTOR_MODEL_DIR"
  hf_cli download "$ANSWER_MODEL_ID" --revision "$ANSWER_MODEL_REVISION" \
    --local-dir "$ANSWER_MODEL_DIR"
}

prepare_requests() {
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.timelens2_quality prepare \
    --source "$INPUT_DIR/source/inference-manifest.json" \
    --dataset-root "$INPUT_DIR/data" \
    --container-media-root "$INPUT_DIR/data/vqa/2025/training/videos_original" \
    --output "$OUTPUT_DIR/quality-requests.json" \
    --expected-source-sha256 "$EXPECTED_SOURCE_SHA256"
}

run_inference() {
  local ffmpeg_executable
  ffmpeg_executable="$("$PYTHON_BIN" -c 'from imageio_ffmpeg import get_ffmpeg_exe; print(get_ffmpeg_exe())')"
  [[ "$ffmpeg_executable" = /* && -x "$ffmpeg_executable" ]] \
    || die "pinned ffmpeg executable is unavailable"
  exec "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.timelens2_quality run \
    --requests "$OUTPUT_DIR/quality-requests.json" \
    --selector-model-dir "$SELECTOR_MODEL_DIR" \
    --answer-model-dir "$ANSWER_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --ffmpeg "$ffmpeg_executable"
}

write_terminal_status() {
  local workload_rc="$1"
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"workload_returncode": int(sys.argv[2])}, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

finalize_and_sync() {
  local workload_rc="$1"
  mkdir -p "$OUTPUT_DIR"
  write_terminal_status "$workload_rc"
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS -print0 | sort -z \
      | xargs -0 sha256sum >SHA256SUMS
  )
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$HF_OUTPUT_URI"
  local sync_rc=$?
  set -e
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

run_workload() {
  validate_environment
  mkdir -p "$OUTPUT_DIR" /dstack/models
  bootstrap_runtime
  hf_cli version
  hf_cli auth whoami
  require_remote_input_and_empty_output
  download_and_verify_input
  prepare_requests
  download_models
  run_inference
}

case "${1:-}" in
  validate)
    validate_environment
    ;;
  run)
    mkdir -p "$OUTPUT_DIR"
    set +e
    (
      set -euo pipefail
      run_workload
    )
    workload_rc=$?
    set -e
    finalize_and_sync "$workload_rc"
    ;;
  *) die "usage: $0 validate|run" ;;
esac
