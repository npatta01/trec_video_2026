#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_RUNTIME_IMAGE_DIGEST="sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_MODEL_ID="MCG-NJU/TimeLens2-2B"
readonly EXPECTED_MODEL_REVISION="90dd30b3eeff3520a30c830eff2209f4ad788ca9"
readonly TRANSFORMERS_VERSION="4.57.6"
readonly QWEN_VL_UTILS_VERSION="0.0.14"
readonly PYAV_VERSION="16.0.1"
readonly ACCELERATE_VERSION="1.12.0"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.26.0"

HF_ACCESS_TOKEN="${HF_TOKEN:-}"
readonly HF_ACCESS_TOKEN
unset HF_TOKEN

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
RUN_MODE="${RUN_MODE:-}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
RUNTIME_IMAGE_DIGEST="${RUNTIME_IMAGE_DIGEST:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
MODEL_DIR="${MODEL_DIR:-/dstack/model}"
REQUESTS_PATH="${REQUESTS_PATH:-/dstack/input/compatibility-requests.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"

die() {
  echo "error: $*" >&2
  exit 2
}

validate_environment() {
  [[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || die "invalid image interpreter"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image interpreter identity mismatch"
  [[ "$RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image mismatch"
  [[ "$RUNTIME_IMAGE_DIGEST" == "$EXPECTED_RUNTIME_IMAGE_DIGEST" ]] \
    || die "runtime image digest mismatch"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" ]] || die "model ID mismatch"
  [[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die "model revision mismatch"
  case "$RUN_MODE" in
    canary)
      [[ "$EXPERIMENT_RUN_ID" == "timelens2-runner-canary-20260730-a1" ]] \
        || die "unexpected canary run ID"
      ;;
    compatibility)
      case "$EXPERIMENT_RUN_ID" in
        timelens2-selector-pilot-20260730-a1|timelens2-selector-pilot-20260730-a2|timelens2-selector-pilot-20260801-a1) ;;
        *) die "unexpected compatibility run ID" ;;
      esac
      [[ -f "$REQUESTS_PATH" ]] || die "label-free request manifest is missing"
      ;;
    *) die "RUN_MODE must be canary or compatibility" ;;
  esac
  local expected_remote
  expected_remote="hf://buckets/Npatta01/trec_video_2026/experiments/${EXPERIMENT_RUN_ID}"
  [[ "$HF_REMOTE_URI" == "$expected_remote" ]] || die "remote URI mismatch"
  [[ "$MODEL_DIR" == /dstack/model ]] || die "model directory mismatch"
  [[ "$OUTPUT_DIR" == /dstack/output ]] || die "output directory mismatch"
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "uv==$UV_VERSION"
  command -v uvx >/dev/null || die "uvx is unavailable"
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: \
    "transformers==$TRANSFORMERS_VERSION" \
    "qwen-vl-utils[decord]==$QWEN_VL_UTILS_VERSION" \
    "av==$PYAV_VERSION" \
    "accelerate==$ACCELERATE_VERSION"
  "$PYTHON_BIN" - "$TRANSFORMERS_VERSION" "$QWEN_VL_UTILS_VERSION" "$PYAV_VERSION" <<'PY'
import importlib.metadata
import sys
import torch

assert importlib.metadata.version("transformers") == sys.argv[1]
assert importlib.metadata.version("qwen-vl-utils") == sys.argv[2]
assert importlib.metadata.version("av") == sys.argv[3]
assert torch.cuda.is_available()
assert torch.version.cuda
assert torch.version.hip is None
PY
}

hf_cli() {
  HF_TOKEN="$HF_ACCESS_TOKEN" uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

remote_prefix_is_empty() {
  local listing
  listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format quiet)" \
    || die "could not verify immutable remote prefix"
  [[ -z "$listing" ]] || die "immutable remote run prefix already exists"
}

download_model() {
  if [[ "$RUN_MODE" == canary ]]; then
    hf_cli download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR" \
      --include '*.json' --include '*.txt' --include '*.jinja' --include '*.model'
  else
    hf_cli download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
  fi
}

write_terminal_status() {
  local workload_rc="$1"
  local sync_attempted="$2"
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" "$sync_attempted" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "workload_returncode": int(sys.argv[2]),
            "artifact_sync_attempted": sys.argv[3] == "true",
        },
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
}

finalize_and_sync() {
  local workload_rc="$1"
  write_terminal_status "$workload_rc" true
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS -print0 | sort -z \
      | xargs -0 sha256sum >SHA256SUMS
  )
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI"
  local sync_rc=$?
  set -e
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

run_python_mode() {
  local mode="$1"
  local argv=(run --mode "$mode" --model-dir "$MODEL_DIR" --output-dir "$OUTPUT_DIR")
  if [[ "$mode" != canary ]]; then
    argv+=(--requests "$REQUESTS_PATH")
  fi
  (exec "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.timelens2_selector "${argv[@]}")
}

run_workload() {
  validate_environment
  mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"
  bootstrap_runtime
  hf_cli version
  hf_cli auth whoami
  remote_prefix_is_empty

  if [[ "$RUN_MODE" == compatibility ]]; then
    chmod -R a-w /dstack/input
    set +e
    run_python_mode "media-preflight"
    local preflight_rc=$?
    set -e
    if (( preflight_rc != 0 )); then
      finalize_and_sync "$preflight_rc"
      return
    fi
  fi

  set +e
  download_model
  local download_rc=$?
  set -e
  if (( download_rc != 0 )); then
    finalize_and_sync "$download_rc"
    return
  fi

  set +e
  run_python_mode "$RUN_MODE"
  local workload_rc=$?
  set -e
  finalize_and_sync "$workload_rc"
}

case "${1:-}" in
  validate) validate_environment ;;
  run) run_workload ;;
  *) die "usage: $0 validate|run" ;;
esac
