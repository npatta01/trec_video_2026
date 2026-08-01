#!/usr/bin/env bash
set -euo pipefail

readonly CUDA_IMAGE="docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
readonly ROCM_IMAGE="docker.io/rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
readonly CUDA_SMOKE_IMAGE="localhost/trec-videochat3-cuda-smoke:runtime-certification"
readonly CUDA_SMOKE_CONTAINERFILE="${REPO_ROOT}/containers/videochat3-cuda-smoke.Containerfile"
ROCM_DEVICE_CHECK_ROOT="${ROCM_DEVICE_CHECK_ROOT:-/}"

usage() {
  echo "usage: $0 {cpu|rocm|local}" >&2
  exit 2
}

run_cpu() {
  podman build --pull=never \
    --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
    --file "${CUDA_SMOKE_CONTAINERFILE}" \
    --tag "${CUDA_SMOKE_IMAGE}" \
    "${REPO_ROOT}"
  podman run --rm --pull=never --network=none \
    -e PYTHONPATH=/workspace/src \
    -v "${REPO_ROOT}/src:/workspace/src:ro" \
    "${CUDA_SMOKE_IMAGE}" \
    python -m trec_video_2026.vqa_experiments.videochat3_runtime_certification cpu-smoke
}

run_rocm() {
  if ! podman image exists "${ROCM_IMAGE}"; then
    echo "advisory skip: cached ROCm image is unavailable" >&2
    return 0
  fi
  if [[ ! -e "${ROCM_DEVICE_CHECK_ROOT%/}/dev/kfd" || ! -e "${ROCM_DEVICE_CHECK_ROOT%/}/dev/dri" ]]; then
    echo "advisory skip: ROCm devices are unavailable" >&2
    return 0
  fi
  podman run --rm --pull=never --network=none \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=keep-groups \
    -e PYTHONPATH=/workspace/src \
    -v "${REPO_ROOT}/src:/workspace/src:ro" \
    "${ROCM_IMAGE}" \
    python -m trec_video_2026.vqa_experiments.videochat3_runtime_certification rocm-smoke
}

case "${1:-}" in
  cpu)
    run_cpu
    ;;
  rocm)
    run_rocm
    ;;
  local)
    run_cpu
    run_rocm
    ;;
  *)
    usage
    ;;
esac
