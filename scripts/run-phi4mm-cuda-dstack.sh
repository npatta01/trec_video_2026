#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_RUNTIME_IMAGE_DIGEST="sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_CONTRACT_SHA256="b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
readonly PHI_TRANSFORMERS_VERSION="4.48.2"
readonly GEMMA_TRANSFORMERS_VERSION="5.14.1"
readonly SCIPY_VERSION="1.15.1"
# Transformers 4.53.2's timm<=1.0.11 optional metadata predates the
# MobileNetV5 encoder required by this Gemma snapshot. These pins match the
# compatible package family already exercised by the frozen ROCm probe.
readonly TIMM_VERSION="1.0.28"
readonly LIBROSA_VERSION="0.11.0"
readonly SAFE_OUTPUT_DIR="/dstack/output"

MODEL_KEY="${MODEL_KEY:-phi4mm}"
case "$MODEL_KEY" in
  phi4mm)
    readonly EXPECTED_RUN_ID="phi4mm-cuda-quality-3-20260724-a4"
    readonly EXPECTED_MODEL_ID="microsoft/Phi-4-multimodal-instruct"
    readonly EXPECTED_MODEL_REVISION="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
    readonly TRANSFORMERS_VERSION="$PHI_TRANSFORMERS_VERSION"
    ;;
  gemma3n)
    readonly EXPECTED_RUN_ID="gemma3n-cuda-quality-3-20260724-a1"
    readonly EXPECTED_MODEL_ID="google/gemma-3n-E4B-it"
    readonly EXPECTED_MODEL_REVISION="c1221e9c62e34a43ab7ffacd1be0ea71f126ef10"
    readonly TRANSFORMERS_VERSION="$GEMMA_TRANSFORMERS_VERSION"
    ;;
  *)
    echo "error: unexpected model key" >&2
    exit 2
    ;;
esac
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/${EXPECTED_RUN_ID}"

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
  mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"
}

bootstrap_artifact_cli() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: \
    "uv==$UV_VERSION"
  command -v uvx >/dev/null || die "uvx is unavailable after bootstrap"
}

bootstrap_runtime() {
  local runtime_packages=(
    "accelerate==1.3.0" \
    "backoff==2.2.1" \
    "peft==0.13.2" \
    "transformers==$TRANSFORMERS_VERSION" \
    "scipy==$SCIPY_VERSION" \
    "soundfile==0.13.1" \
    "pillow==11.1.0" \
    "torchvision==0.20.1"
  )
  if [[ "$MODEL_KEY" == "gemma3n" ]]; then
    runtime_packages+=(
      "timm==$TIMM_VERSION"
      "librosa==$LIBROSA_VERSION"
    )
  fi
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: \
    "${runtime_packages[@]}"
  "$PYTHON_BIN" - "$MODEL_KEY" "$TRANSFORMERS_VERSION" <<'PY'
import importlib.metadata
import sys
import torch
import transformers

model_key = sys.argv[1]
expected_transformers = sys.argv[2]
assert importlib.metadata.version("transformers") == expected_transformers
assert importlib.metadata.version("scipy") == "1.15.1"
if model_key == "gemma3n":
    assert importlib.metadata.version("timm") == "1.0.28"
    assert importlib.metadata.version("librosa") == "0.11.0"
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

authenticate() {
  hf_cli version
  hf_cli auth whoami
}

remote_prefix_is_empty() {
  local listing
  local prefix_rc
  if listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format quiet)"; then
    prefix_rc=0
  else
    prefix_rc=$?
  fi
  if ! (( prefix_rc == 0 )); then
    echo "could not verify immutable remote prefix (rc=$prefix_rc)" >&2
    return "$prefix_rc"
  fi
  if [[ -n "$listing" ]]; then
    echo "immutable remote run prefix already exists" >&2
    return 3
  fi
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

preflight_model_import() {
  if [[ "$MODEL_KEY" == "gemma3n" ]]; then
    env -u HF_TOKEN "$PYTHON_BIN" - \
      "$MODEL_DIR" \
      "$CONTRACT_PATH" \
      "$DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import timm
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from trec_video_2026.vqa_experiments.omni_contract import (
    validate_inference_evidence,
)
from trec_video_2026.vqa_experiments.omni_probe import resolve_example_paths
from trec_video_2026.vqa_experiments.omni_runtime import gemma_messages

model_dir = Path(sys.argv[1])
contract_path = Path(sys.argv[2])
data_root = Path(sys.argv[3])
if not timm.is_model("mobilenetv5_300m_enc"):
    raise ValueError("timm lacks the Gemma 3n MobileNetV5 encoder")
processor = AutoProcessor.from_pretrained(
    model_dir,
    local_files_only=True,
)
contract = json.loads(contract_path.read_text(encoding="utf-8"))
validate_inference_evidence(contract, data_root, verify_files=True)
example = resolve_example_paths(contract["examples"][0], data_root)
processed = processor.apply_chat_template(
    gemma_messages(example),
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
for required_key in ("input_features", "pixel_values"):
    if required_key not in processed:
        raise ValueError(f"Gemma preflight omitted {required_key}")
print(
    json.dumps(
        {
            "model_class": Gemma3nForConditionalGeneration.__name__,
            "processor_class": processor.__class__.__name__,
            "preflight_input_keys": sorted(processed),
        },
        sort_keys=True,
    )
)
PY
    return
  fi
  env -u HF_TOKEN "$PYTHON_BIN" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

model_dir = Path(sys.argv[1])
processor = AutoProcessor.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)
model_class = get_class_from_dynamic_module(
    "modeling_phi4mm.Phi4MMForCausalLM",
    model_dir,
    local_files_only=True,
)
print(
    json.dumps(
        {
            "model_class": model_class.__name__,
            "processor_class": processor.__class__.__name__,
        },
        sort_keys=True,
    )
)
PY
}

write_terminal_status() {
  local workload_rc="$1"
  local failure_stage="$2"
  local sync_attempted="$3"
  "$PYTHON_BIN" - \
    "$OUTPUT_DIR/terminal-status.json" \
    "$workload_rc" \
    "$failure_stage" \
    "$sync_attempted" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "workload_returncode": int(sys.argv[2]),
            "failure_stage": sys.argv[3],
            "artifact_sync_attempted": sys.argv[4] == "1",
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
  local failure_stage="$2"
  local remote_prefix_verified="$3"
  write_terminal_status \
    "$workload_rc" \
    "$failure_stage" \
    "$remote_prefix_verified"
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum >SHA256SUMS
  )
  if (( remote_prefix_verified == 0 )); then
    return "$workload_rc"
  fi
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI"
  local sync_rc=$?
  set -e
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

run_stage() {
  local stage="$1"
  shift
  failure_stage="$stage"
  set +e
  (
    set -e
    "$@"
  ) >"$OUTPUT_DIR/${stage}.stdout.log" \
    2>"$OUTPUT_DIR/${stage}.stderr.log"
  workload_rc=$?
  set -e
}

run_inference() {
  env -u HF_TOKEN "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.omni_probe \
    --model "$MODEL_KEY" \
    --model-path "$MODEL_DIR" \
    --contract "$CONTRACT_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR/results" \
    --runtime-image "$RUNTIME_IMAGE" \
    --runtime-image-digest "$RUNTIME_IMAGE_DIGEST" \
    --run-id "$EXPERIMENT_RUN_ID" \
    --expected-contract-sha256 "$EXPECTED_CONTRACT"
}

run_probe() {
  local workload_rc=0
  local failure_stage="environment_validation"
  local remote_prefix_verified=0

  mkdir -p "$SAFE_OUTPUT_DIR"
  run_stage environment_validation validate_environment
  if (( workload_rc == 0 )); then
    run_stage artifact_cli_bootstrap bootstrap_artifact_cli
  fi
  if (( workload_rc == 0 )); then
    run_stage authentication authenticate
  fi
  if (( workload_rc == 0 )); then
    run_stage remote_prefix_check remote_prefix_is_empty
    if (( workload_rc == 0 )); then
      remote_prefix_verified=1
    fi
  fi
  if (( workload_rc == 0 )); then
    run_stage input_preflight preflight_inputs
  fi
  if (( workload_rc == 0 )); then
    run_stage runtime_bootstrap bootstrap_runtime
  fi
  if (( workload_rc == 0 )); then
    run_stage model_download download_model
  fi
  if (( workload_rc == 0 )); then
    run_stage model_import_preflight preflight_model_import
  fi
  if (( workload_rc == 0 )); then
    run_stage inference run_inference
  fi

  finalize_and_sync "$workload_rc" "$failure_stage" "$remote_prefix_verified"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    run)
      run_probe
      ;;
    *)
      die "usage: $0 run"
      ;;
  esac
fi
