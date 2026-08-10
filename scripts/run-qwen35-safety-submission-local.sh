#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly DATA_ROOT="$EXPECTED_DATA_TARGET"
readonly RUN_ROOT="$DATA_ROOT/vqa/submissions/qwen35-safety-20260810-a1"
readonly REQUESTS="$RUN_ROOT/requests.json"
readonly OUTPUT="$RUN_ROOT/predictions.raw.json"
readonly RUN_ID="qwen35-safety-20260810-a1"
readonly IMAGE="localhost/trec-vqa-omni-rocm@sha256:b1f97c47769a8a2f32a5ac9a18cc62d2aab57ed32c66720775407f6213ebf96f"
readonly HF_CACHE="/home/npatta01/.cache/huggingface"
readonly MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ -L "$REPO_ROOT/data" ]] || die "repository data path must be a symlink"
[[ "$(readlink -f "$REPO_ROOT/data")" == "$EXPECTED_DATA_TARGET" ]] \
  || die "repository data symlink target mismatch"
[[ -f "$RUN_ROOT/manifest.json" ]] || die "frozen manifest is missing"
[[ -f "$REQUESTS" ]] || die "prepared request file is missing"
podman image exists "$IMAGE" || die "pinned ROCm image is missing"
readonly MODEL_REF="$HF_CACHE/hub/models--Qwen--Qwen3.5-4B/refs/main"
[[ -f "$MODEL_REF" ]] || die "pinned Qwen cache is missing"
[[ "$(<"$MODEL_REF")" == "$MODEL_REVISION" ]] \
  || die "cached Qwen revision does not match the frozen revision"
[[ -d "$HF_CACHE/hub/models--Qwen--Qwen3.5-4B/snapshots/$MODEL_REVISION" ]] \
  || die "pinned Qwen snapshot is incomplete"

exec podman run --rm \
  --network=none \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=keep-groups \
  --security-opt label=disable \
  --shm-size=16g \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/src \
  -e HF_HOME=/cache \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e VQA_RUNTIME_IMAGE="$IMAGE" \
  -v "$HF_CACHE:/cache:ro" \
  -v "$DATA_ROOT:$DATA_ROOT:rw" \
  -v "$REPO_ROOT/src:/workspace/src:ro" \
  "$IMAGE" \
  python -m trec_video_2026.vqa_safety_runner \
    --requests "$REQUESTS" \
    --output "$OUTPUT" \
    --run-id "$RUN_ID"
