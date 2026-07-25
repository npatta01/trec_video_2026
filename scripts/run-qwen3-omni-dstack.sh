#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUN_ID="qwen3-omni-compatibility-3-20260724-a4"
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/${EXPECTED_RUN_ID}"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/qwenllm/qwen3-omni@sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
readonly EXPECTED_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Thinking"
readonly EXPECTED_MODEL_REVISION="2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
readonly EXPECTED_CONTRACT_SHA256="b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
EXPECTED_CONTRACT="${EXPECTED_CONTRACT:-}"
CONTRACT_PATH="${CONTRACT_PATH:-/dstack/input/contract.json}"
DATA_ROOT="${DATA_ROOT:-/dstack/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"

die() {
  echo "error: $*" >&2
  exit 2
}

validate_environment() {
  [[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] \
    || die "PYTHON_BIN must name the exact absolute image interpreter"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image interpreter identity mismatch"
  [[ "$EXPERIMENT_RUN_ID" == "$EXPECTED_RUN_ID" ]] || die "unexpected run ID"
  [[ "$HF_REMOTE_URI" == "$EXPECTED_REMOTE_URI" ]] || die "unexpected remote URI"
  [[ "$RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image mismatch"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" ]] || die "model ID mismatch"
  [[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die "model revision mismatch"
  [[ "$EXPECTED_CONTRACT" == "$EXPECTED_CONTRACT_SHA256" ]] \
    || die "contract digest mismatch"
  [[ "$CONTRACT_PATH" == /dstack/input/contract.json ]] \
    || die "contract path mismatch"
  [[ "$DATA_ROOT" == /dstack/input ]] || die "data root mismatch"
  [[ "$OUTPUT_DIR" == /dstack/output ]] || die "output path mismatch"
  [[ -f "$CONTRACT_PATH" ]] || die "contract is missing"
  chmod -R a-w "$DATA_ROOT"
  mkdir -p "$OUTPUT_DIR"
}

bootstrap_uv() {
  "$PYTHON_BIN" -m pip install \
    --no-deps \
    "uv==$UV_VERSION"
  [[ "$("$PYTHON_BIN" -c \
    'import importlib.metadata; print(importlib.metadata.version("uv"))')" == "$UV_VERSION" ]] \
    || die "uv bootstrap attestation failed"
  command -v uvx >/dev/null || die "uvx is unavailable after bootstrap"
}

hf_cli() {
  uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

remote_prefix_is_empty() {
  local listing
  set +e
  listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format json 2>/tmp/hf-list.stderr)"
  local rc=$?
  set -e
  (( rc == 0 )) || die "could not verify immutable remote prefix (rc=$rc)"
  [[ -z "$listing" ]]
}

write_terminal_status() {
  local workload_rc="$1"
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "workload_returncode": int(sys.argv[2]),
            "artifact_sync_attempted": True,
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

finalize_and_sync() {
  local workload_rc="$1"
  write_terminal_status "$workload_rc"
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum >SHA256SUMS
  )

  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI"
  local sync_rc=$?
  set -e
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

run_probe() {
  validate_environment
  bootstrap_uv
  hf_cli version
  hf_cli auth whoami
  remote_prefix_is_empty || die "immutable remote run prefix already exists"

  set +e
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.qwen3_omni_dstack \
    --contract "$CONTRACT_PATH" \
    --data-root "$DATA_ROOT" \
    --preflight >"$OUTPUT_DIR/preflight.stdout.json"
  local preflight_rc=$?
  set -e
  if (( preflight_rc != 0 )); then
    finalize_and_sync "$preflight_rc" || :
    return "$preflight_rc"
  fi

  set +e
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.qwen3_omni_dstack \
    --contract "$CONTRACT_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$EXPERIMENT_RUN_ID" >"$OUTPUT_DIR/workload.stdout.json"
  local workload_rc=$?
  set -e

  finalize_and_sync "$workload_rc"
}

case "${1:-}" in
  run)
    run_probe
    ;;
  *)
    die "usage: $0 run"
    ;;
esac
