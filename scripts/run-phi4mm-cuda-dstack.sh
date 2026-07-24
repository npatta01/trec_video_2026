#!/usr/bin/env bash
set -euo pipefail

# HISTORICAL FAILED-RUN RECORD: this preserves the a3 dependency envelope,
# including the missing SciPy dependency that stopped model loading. It is not
# a reusable a4 runner, and its immutable remote prefix is already occupied.
readonly EXPECTED_RUN_ID="phi4mm-cuda-quality-3-20260724-a3"
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/${EXPECTED_RUN_ID}"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_RUNTIME_IMAGE_DIGEST="sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_MODEL_ID="microsoft/Phi-4-multimodal-instruct"
readonly EXPECTED_MODEL_REVISION="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
readonly EXPECTED_CONTRACT_SHA256="b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
readonly TRANSFORMERS_VERSION="4.48.2"

HF_ACCESS_TOKEN="${HF_TOKEN:-}"
readonly HF_ACCESS_TOKEN
unset HF_TOKEN

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
RUNTIME_IMAGE_DIGEST="${RUNTIME_IMAGE_DIGEST:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
EXPECTED_CONTRACT="${EXPECTED_CONTRACT:-}"
CONTRACT_PATH="${CONTRACT_PATH:-/dstack/input/contract.json}"
DATA_ROOT="${DATA_ROOT:-/dstack/input}"
MODEL_DIR="${MODEL_DIR:-/dstack/model}"
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
  [[ "$RUNTIME_IMAGE_DIGEST" == "$EXPECTED_RUNTIME_IMAGE_DIGEST" ]] \
    || die "runtime image digest mismatch"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" ]] || die "model ID mismatch"
  [[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die "model revision mismatch"
  [[ "$EXPECTED_CONTRACT" == "$EXPECTED_CONTRACT_SHA256" ]] \
    || die "contract digest mismatch"
  [[ "$DATA_ROOT" == "/dstack/input" ]] || die "unexpected data root"
  [[ "$CONTRACT_PATH" == "/dstack/input/contract.json" ]] \
    || die "unexpected contract path"
  [[ "$MODEL_DIR" == "/dstack/model" ]] || die "unexpected model directory"
  [[ "$OUTPUT_DIR" == "/dstack/output" ]] || die "unexpected output directory"
  [[ -f "$CONTRACT_PATH" ]] || die "contract is missing"
  chmod -R a-w "$DATA_ROOT"
  mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: \
    "uv==$UV_VERSION" \
    "accelerate==1.3.0" \
    "backoff==2.2.1" \
    "peft==0.13.2" \
    "transformers==$TRANSFORMERS_VERSION" \
    "soundfile==0.13.1" \
    "pillow==11.1.0" \
    "torchvision==0.20.1"
  command -v uvx >/dev/null || die "uvx is unavailable after bootstrap"
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import torch
import transformers

assert importlib.metadata.version("transformers") == "4.48.2"
assert torch.cuda.is_available(), "NVIDIA CUDA accelerator is unavailable"
assert torch.version.hip is None, "ROCm runtime is not permitted for this run"
assert torch.version.cuda, "CUDA runtime version is missing"
assert torch.cuda.is_bf16_supported(), "GPU does not support bfloat16"
print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
)
PY
}

hf_cli() {
  HF_TOKEN="$HF_ACCESS_TOKEN" uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
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

preflight_inputs() {
  "$PYTHON_BIN" - "$CONTRACT_PATH" "$DATA_ROOT" "$EXPECTED_CONTRACT" <<'PY'
import json
import sys
from pathlib import Path

from trec_video_2026.vqa_experiments.omni_contract import (
    contract_sha256,
    validate_inference_evidence,
)

contract_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])
expected = sys.argv[3]
contract = json.loads(contract_path.read_text(encoding="utf-8"))
validate_inference_evidence(contract, data_root, verify_files=True)
actual = contract_sha256(contract)
if actual != expected:
    raise ValueError(f"contract digest mismatch: expected {expected}, got {actual}")
print(json.dumps({"contract_sha256": actual, "rows": len(contract["examples"])}))
PY
}

download_model() {
  hf_cli download "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_DIR"
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
  bootstrap_runtime
  hf_cli version
  hf_cli auth whoami
  remote_prefix_is_empty || die "immutable remote run prefix already exists"

  set +e
  preflight_inputs >"$OUTPUT_DIR/preflight.stdout.json"
  local preflight_rc=$?
  set -e
  if (( preflight_rc != 0 )); then
    finalize_and_sync "$preflight_rc" || :
    return "$preflight_rc"
  fi

  download_model
  set +e
  env -u HF_TOKEN "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.omni_probe \
    --model phi4mm \
    --model-path "$MODEL_DIR" \
    --contract "$CONTRACT_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR/results" \
    --runtime-image "$RUNTIME_IMAGE" \
    --runtime-image-digest "$RUNTIME_IMAGE_DIGEST" \
    --run-id "$EXPERIMENT_RUN_ID" \
    --expected-contract-sha256 "$EXPECTED_CONTRACT" \
    >"$OUTPUT_DIR/workload.stdout.json"
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
