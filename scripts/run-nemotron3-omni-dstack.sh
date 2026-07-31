#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
readonly EXPECTED_RUNTIME_IMAGE_DIGEST="sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
readonly EXPECTED_MODEL_ID="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
readonly EXPECTED_MODEL_REVISION="dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
readonly OUTPUT_DIR="/dstack/output"
readonly INPUT_DIR="/dstack/input"

MODE="${1:-}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
HF_INPUT_URI="${HF_INPUT_URI:-$HF_REMOTE_URI}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
RUNTIME_IMAGE_DIGEST="${RUNTIME_IMAGE_DIGEST:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
HF_ACCESS_TOKEN="${HF_TOKEN:-}"
readonly HF_ACCESS_TOKEN
unset HF_TOKEN
failure_stage="initializing"
output_prefix_verified="false"
finalized="false"
ACTIVE_CONTRACT="$INPUT_DIR/contract.json"
ACTIVE_MEDIA_ROOT="$INPUT_DIR"

die() {
  echo "error: $*" >&2
  exit 2
}

validate_identity() {
  case "$MODE:$EXPERIMENT_RUN_ID" in
    canary:nemotron3-omni-runner-canary-20260730-a1|canary:nemotron3-omni-runner-canary-20260730-a2) ;;
    pilot:nemotron3-omni-pilot-20260730-a1) ;;
    codec-sentinel:nemotron3-omni-codec-sentinel-20260730-b1) ;;
    pilot:nemotron3-omni-pilot-20260730-b1) ;;
    canary:nemotron3-omni-expanded12-canary-20260730-c1) ;;
    expanded-pilot:nemotron3-omni-expanded12-20260730-c1) ;;
    canary:nemotron3-omni-expanded12-canary-20260731-d1) ;;
    expanded-pilot:nemotron3-omni-expanded12-20260731-d1) ;;
    *) die "unexpected immutable mode/run identity" ;;
  esac
  [[ "$HF_REMOTE_URI" == "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$EXPERIMENT_RUN_ID" ]] \
    || die "unexpected remote URI"
  case "$EXPERIMENT_RUN_ID" in
    nemotron3-omni-codec-sentinel-20260730-b1|nemotron3-omni-pilot-20260730-b1)
      [[ "$HF_INPUT_URI" == "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-pilot-20260730-b1" ]] \
        || die "unexpected input URI"
      ;;
    *)
      [[ "$HF_INPUT_URI" == "$HF_REMOTE_URI" ]] || die "unexpected input URI"
      ;;
  esac
  [[ "$RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image mismatch"
  [[ "$RUNTIME_IMAGE_DIGEST" == "$EXPECTED_RUNTIME_IMAGE_DIGEST" ]] \
    || die "runtime image digest mismatch"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" ]] || die "model ID mismatch"
  [[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die "model revision mismatch"
  [[ -n "$HF_ACCESS_TOKEN" ]] || die "HF_TOKEN secret is unavailable"
  [[ "$(uname -m)" == "x86_64" ]] || die "amd64/x86_64 runtime is required"
  [[ -x "$PYTHON_BIN" ]] || die "exact image interpreter is unavailable"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image interpreter identity mismatch"
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "vllm[audio]==0.20.0" "uv==$UV_VERSION"
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import platform
import torch

vllm_version = importlib.metadata.version("vllm")
if vllm_version != "0.20.0":
    raise RuntimeError(f"unexpected vLLM version: {vllm_version}")
assert platform.machine() == "x86_64", "amd64 image is required"
assert torch.cuda.is_available(), "CUDA accelerator is unavailable"
assert torch.version.hip is None, "ROCm is forbidden"
assert torch.version.cuda, "CUDA runtime is unavailable"
print({"vllm": vllm_version, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PY
  VLLM_BIN="$("$PYTHON_BIN" - <<'PY'
import os
import sysconfig
print(os.path.join(sysconfig.get_path("scripts"), "vllm"))
PY
)"
  readonly VLLM_BIN
  [[ "$VLLM_BIN" = /* && -x "$VLLM_BIN" ]] || die "vLLM executable is unavailable"
  command -v uvx >/dev/null || die "uvx is unavailable"
}

hf_cli() {
  HF_TOKEN="$HF_ACCESS_TOKEN" uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

remote_outputs_are_empty() {
  local listing
  listing="$(hf_cli buckets list "$HF_REMOTE_URI/outputs" --format quiet)" \
    || die "could not inspect immutable output prefix"
  [[ -z "$listing" ]] || die "immutable output prefix already exists"
}

create_canary_inputs() {
  command -v ffmpeg >/dev/null || die "ffmpeg is required for public synthetic AV"
  mkdir -p "$INPUT_DIR/media"
  local colors=(red green blue)
  local frequencies=(440 554 659)
  local media_args=()
  local index output
  for index in 0 1 2; do
    output="$INPUT_DIR/media/canary-$((index + 1))-${colors[$index]}.mkv"
    ffmpeg -nostdin -hide_banner -loglevel error \
      -f lavfi -i "color=c=${colors[$index]}:s=320x240:r=8:d=2" \
      -f lavfi -i "sine=frequency=${frequencies[$index]}:sample_rate=16000:duration=2" \
      -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$output"
    "$PYTHON_BIN" - "$output" <<'PY'
import json
import subprocess
import sys

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", sys.argv[1]],
    check=True,
    capture_output=True,
    text=True,
)
stream_types = sorted(stream["codec_type"] for stream in json.loads(probe.stdout)["streams"])
if stream_types != ["audio", "video"]:
    raise RuntimeError(f"synthetic canary must contain audio and video: {stream_types}")
PY
    media_args+=(--media "$output")
  done
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot \
    build-canary-contract "${media_args[@]}" >"$INPUT_DIR/contract.json"
}

download_pilot_inputs() {
  local ids=(IRTTtnSRTyY IWSPAfCKtV8 MdMKHhFFqJQ OzNdzgQi1CA EMI_anpHII4 LusRurRrmf0)
  if [[ "$MODE" == "expanded-pilot" ]]; then
    ids=(NN7G28sBOlY NUmb2P3Bzu8 IRTTtnSRTyY IWSPAfCKtV8 MdMKHhFFqJQ OzNdzgQi1CA CH8Dp7EgW4I EtNOkUJd5w8 Gp4LVTn3YbY KgOvCPOX-Ms EMI_anpHII4 LusRurRrmf0)
  fi
  local id destination
  mkdir -p "$INPUT_DIR/vqa/2025/training/videos_original"
  hf_cli buckets cp "$HF_INPUT_URI/inputs/contract.json" "$INPUT_DIR/contract.json"
  hf_cli buckets cp "$HF_INPUT_URI/inputs/INPUTS_COMPLETE.json" "$INPUT_DIR/INPUTS_COMPLETE.json"
  for id in "${ids[@]}"; do
    destination="$INPUT_DIR/vqa/2025/training/videos_original/$id.mkv"
    hf_cli buckets cp "$HF_INPUT_URI/inputs/media/$id.mkv" "$destination"
  done
}

preflight_inputs() {
  local media_root="$INPUT_DIR"
  local args=(
    validate-inputs
    --contract "$INPUT_DIR/contract.json"
    --media-root "$media_root"
  )
  if [[ "$MODE" == "canary" ]]; then
    media_root="$INPUT_DIR/media"
    args=(
      validate-inputs
      --contract "$INPUT_DIR/contract.json"
      --media-root "$media_root"
    )
  else
    args+=(--marker "$INPUT_DIR/INPUTS_COMPLETE.json")
  fi
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot \
    "${args[@]}"
}

normalize_pilot_inputs() {
  command -v ffmpeg >/dev/null || die "ffmpeg is required for normalization"
  command -v ffprobe >/dev/null || die "ffprobe is required for normalization"
  local source_root="$INPUT_DIR/vqa/2025/training/videos_original"
  local normalized_root="$INPUT_DIR/normalized"
  local ids=(IRTTtnSRTyY IWSPAfCKtV8 MdMKHhFFqJQ OzNdzgQi1CA EMI_anpHII4 LusRurRrmf0)
  if [[ "$MODE" == "expanded-pilot" ]]; then
    ids=(NN7G28sBOlY NUmb2P3Bzu8 IRTTtnSRTyY IWSPAfCKtV8 MdMKHhFFqJQ OzNdzgQi1CA CH8Dp7EgW4I EtNOkUJd5w8 Gp4LVTn3YbY KgOvCPOX-Ms EMI_anpHII4 LusRurRrmf0)
  fi
  local id source output
  mkdir -p "$normalized_root"
  for id in "${ids[@]}"; do
    source="$source_root/$id.mkv"
    output="$normalized_root/$id.mp4"
    ffmpeg -nostdin -hide_banner -loglevel error -xerror -n \
      -i "$source" \
      -map 0:v:0 -map 0:a:0 -map_metadata -1 -map_chapters -1 \
      -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -threads 1 \
      -c:a aac -b:a 192k -ar 48000 -ac 2 \
      -movflags +faststart -shortest "$output"
    "$PYTHON_BIN" - "$source" "$output" <<'PY'
import json
import subprocess
import sys

def probe(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name:format=duration", "-of", "json", path
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)

source = probe(sys.argv[1])
derived = probe(sys.argv[2])
video_codecs = sorted(
    stream["codec_name"] for stream in derived["streams"]
    if stream["codec_type"] == "video"
)
audio_codecs = sorted(
    stream["codec_name"] for stream in derived["streams"]
    if stream["codec_type"] == "audio"
)
if video_codecs != ["h264"]:
    raise RuntimeError(f"normalized video codec mismatch: {video_codecs}")
if audio_codecs != ["aac"]:
    raise RuntimeError(f"normalized audio codec mismatch: {audio_codecs}")
source_duration = float(source["format"]["duration"])
derived_duration = float(derived["format"]["duration"])
if abs(source_duration - derived_duration) > 0.5:
    raise RuntimeError(
        f"normalized duration drift: source={source_duration}, derived={derived_duration}"
    )
PY
    ffmpeg -nostdin -hide_banner -loglevel error -xerror \
      -i "$output" -map 0:v:0 -f null -
    ffmpeg -nostdin -hide_banner -loglevel error -xerror \
      -i "$output" -map 0:a:0 -f null -
  done
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot \
    build-normalized-contract \
    --source-contract "$INPUT_DIR/contract.json" \
    --source-media-root "$INPUT_DIR" \
    --normalized-media-root "$normalized_root" \
    --mode "$MODE" \
    --output "$INPUT_DIR/normalized-contract.json"
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot \
    validate-inputs \
    --contract "$INPUT_DIR/normalized-contract.json" \
    --media-root "$normalized_root"
  ACTIVE_CONTRACT="$INPUT_DIR/normalized-contract.json"
  ACTIVE_MEDIA_ROOT="$normalized_root"
}

start_server() {
  mkdir -p "$OUTPUT_DIR"
  "$VLLM_BIN" serve "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 32768 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --allowed-local-media-path /dstack/input \
    --limit-mm-per-prompt '{"video":1,"audio":1}' \
    --media-io-kwargs '{"video":{"fps":1,"num_frames":64}}' \
    --video-pruning-rate 0.5 \
    --kv-cache-dtype fp8 \
    --reasoning-parser nemotron_v3 \
    >"$OUTPUT_DIR/server.log" 2>&1 &
  server_pid="$!"
  readonly server_pid
}

stop_server() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    wait "$server_pid" || true
  fi
}

wait_for_server() {
  local attempt
  for attempt in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" || true
      die "vLLM server exited before readiness"
    fi
    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
with urlopen("http://127.0.0.1:8000/v1/models", timeout=2) as response:
    assert response.status == 200
PY
    then
      return 0
    fi
    sleep 5
  done
  die "vLLM server readiness timed out"
}

run_foreground_wrapper() {
  local media_root="$ACTIVE_MEDIA_ROOT"
  if [[ "$MODE" == "canary" ]]; then
    media_root="$INPUT_DIR/media"
  fi
  set +e
  "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot run \
    --contract "$ACTIVE_CONTRACT" \
    --media-root "$media_root" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$EXPERIMENT_RUN_ID"
  workload_rc=$?
  set -e
}

write_technical_error() {
  local exit_status="$1"
  [[ ! -e "$OUTPUT_DIR/technical-error.json" ]] || return 0
  "$PYTHON_BIN" - "$OUTPUT_DIR/technical-error.json" "$EXPERIMENT_RUN_ID" \
    "$failure_stage" "$exit_status" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "run_id": sys.argv[2],
            "failure_stage": sys.argv[3],
            "exit_status": int(sys.argv[4]),
            "credentials_recorded": False,
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
}

on_exit() {
  local exit_status="$?"
  trap - EXIT
  if (( exit_status != 0 )) && [[ "$finalized" != "true" ]]; then
    stop_server
    mkdir -p "$OUTPUT_DIR"
    write_technical_error "$exit_status"
    if [[ "$output_prefix_verified" == "true" ]]; then
      set +e
      finalize_and_sync "$exit_status"
      set -e
    fi
  fi
  exit "$exit_status"
}

finalize_and_sync() {
  local workload_status="$1"
  stop_server
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum -c SHA256SUMS
    chmod a-w ./*
  )
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI/outputs"
  sync_rc=$?
  set -e
  (( workload_status == 0 )) || return "$workload_status"
  return "$sync_rc"
}

main() {
  failure_stage="identity"
  validate_identity
  mkdir -p "$OUTPUT_DIR"
  trap on_exit EXIT
  failure_stage="runtime-bootstrap"
  bootstrap_runtime
  failure_stage="artifact-authentication"
  hf_cli version
  hf_cli auth whoami
  failure_stage="immutable-output-check"
  remote_outputs_are_empty
  output_prefix_verified="true"
  failure_stage="input-preparation"
  if [[ "$MODE" == "canary" ]]; then
    create_canary_inputs
  else
    download_pilot_inputs
  fi
  preflight_inputs
  case "$EXPERIMENT_RUN_ID" in
    nemotron3-omni-codec-sentinel-20260730-b1|nemotron3-omni-pilot-20260730-b1|nemotron3-omni-expanded12-20260730-c1|nemotron3-omni-expanded12-20260731-d1)
      failure_stage="media-normalization"
      normalize_pilot_inputs
      ;;
  esac
  failure_stage="server-startup"
  start_server
  wait_for_server
  failure_stage="foreground-wrapper"
  run_foreground_wrapper
  failure_stage="artifact-finalization"
  finalized="true"
  finalize_and_sync "$workload_rc"
}

main "$@"
