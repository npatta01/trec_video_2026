#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUN_ID="qwen3-omni-quality-10-20260725-a1"
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-quality-10-20260725-a1"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/qwenllm/qwen3-omni@sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
readonly EXPECTED_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Thinking"
readonly EXPECTED_MODEL_REVISION="2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
readonly EXPECTED_CONTRACT_SHA256="df016767b1a072770149fdc06fa47dfe7f17930b6c3145af4e92f55d58f4ddeb"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"

readonly PYTHON_BIN="/usr/bin/python3"
UVX_BIN=""
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
EXPECTED_CONTRACT="${EXPECTED_CONTRACT:-}"
CONTRACT_PATH="${CONTRACT_PATH:-/dstack/input/contract.json}"
DATA_ROOT="${DATA_ROOT:-/dstack/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
readonly TRANSPORT_CHUNK_DIR="/dstack/input/.transport/frame-11"

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
}

reconstruct_oversized_frame() {
  "$PYTHON_BIN" -m \
    trec_video_2026.vqa_experiments.qwen3_omni_quality_transport reconstruct \
    --data-root "$DATA_ROOT" \
    --chunk-directory "$TRANSPORT_CHUNK_DIR" \
    --contract "$CONTRACT_PATH" \
    || die "oversized frame reconstruction failed"
  chmod -R a-w "$DATA_ROOT"
  mkdir -p "$OUTPUT_DIR"
}

bootstrap_uv() {
  "$PYTHON_BIN" -m pip install --no-deps "uv==$UV_VERSION"
  [[ "$("$PYTHON_BIN" -c \
    'import importlib.metadata; print(importlib.metadata.version("uv"))')" == "$UV_VERSION" ]] \
    || die "uv bootstrap attestation failed"
  if ! UVX_BIN="$("$PYTHON_BIN" - <<'PY'
import importlib.metadata
import sysconfig
from pathlib import Path

distribution = importlib.metadata.distribution("uv")
scripts_directory = Path(sysconfig.get_path("scripts")).resolve()
uvx_path = (scripts_directory / "uvx").resolve()
installed_paths = {
    distribution.locate_file(path).resolve() for path in distribution.files or ()
}
if uvx_path not in installed_paths:
    raise RuntimeError("pinned uv distribution does not own its uvx entry point")
print(uvx_path)
PY
  )"; then
    die "could not resolve the pinned uvx entry point"
  fi
  [[ "$UVX_BIN" = /* && -x "$UVX_BIN" ]] \
    || die "pinned uvx entry point is unavailable"
}

hf_cli() {
  "$UVX_BIN" --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

remote_prefix_is_empty() {
  local listing
  set +e
  listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format json 2>/tmp/hf-list.stderr)"
  local rc=$?
  set -e
  (( rc == 0 )) || die "could not verify immutable remote prefix (rc=$rc)"
  [[ -z "$listing" || "$listing" == "[]" ]]
}

uploaded_contract_sha256() {
  "$PYTHON_BIN" - "$CONTRACT_PATH" <<'PY'
import hashlib
import json
import sys

contract = json.loads(open(sys.argv[1], encoding="utf-8").read())
canonical = json.dumps(
    contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
PY
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
  reconstruct_oversized_frame
  bootstrap_uv
  hf_cli version
  hf_cli auth whoami
  remote_prefix_is_empty || die "immutable remote run prefix already exists"
  local actual_contract_sha256
  actual_contract_sha256="$(uploaded_contract_sha256)" \
    || die "could not calculate uploaded contract digest"
  [[ "$actual_contract_sha256" == "$EXPECTED_CONTRACT_SHA256" ]] \
    || die "uploaded contract digest does not match approved digest"

  set +e
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.qwen3_omni_quality_dstack \
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
