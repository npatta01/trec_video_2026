#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {phi4mm|gemma3n} [run-id]" >&2
  exit 2
fi

MODEL_KEY=$1
RUN_ID=${2:-"${MODEL_KEY}-$(date -u +%Y%m%dT%H%M%SZ)"}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
DATA_ROOT=$(readlink -f "${REPO_ROOT}/data")
HF_CACHE_ROOT=${HF_HUB_CACHE:-/home/npatta01/.cache/huggingface/hub}
RUNTIME_IMAGE_DIGEST=sha256:cbda16214c953bcd2bd2a7bf65d26561ac406b0f1fbab2f719b25baab1d757f4
RUNTIME_IMAGE=localhost/trec-vqa-omni-rocm@sha256:cbda16214c953bcd2bd2a7bf65d26561ac406b0f1fbab2f719b25baab1d757f4
CONTRACT_REL=vqa/benchmarks/omni-compatibility-3/contract.json
EXPECTED_CONTRACT_SHA256=b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "run-id must contain only ASCII letters, digits, dots, underscores, or hyphens and start with a letter or digit" >&2
  exit 2
fi

if [[ "${DATA_ROOT}" != "/home/npatta01/data/competitions/trec_video_2026/data" ]]; then
  echo "data symlink does not resolve to the canonical shared dataset" >&2
  exit 1
fi
if [[ ! -f "${DATA_ROOT}/${CONTRACT_REL}" ]]; then
  echo "missing frozen contract: ${DATA_ROOT}/${CONTRACT_REL}" >&2
  exit 1
fi
CONTRACT_SIDECAR="${DATA_ROOT}/${CONTRACT_REL}.sha256"
EXPECTED_SIDECAR="${EXPECTED_CONTRACT_SHA256}  contract.json"
if [[ ! -f "${CONTRACT_SIDECAR}" ]] || [[ "$(<"${CONTRACT_SIDECAR}")" != "${EXPECTED_SIDECAR}" ]]; then
  echo "frozen contract sidecar does not match expected digest" >&2
  exit 1
fi

case "${MODEL_KEY}" in
  phi4mm)
    MODEL_SNAPSHOT=models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81
    ;;
  gemma3n)
    MODEL_SNAPSHOT=models--google--gemma-3n-E4B-it/snapshots/c1221e9c62e34a43ab7ffacd1be0ea71f126ef10
    ;;
  *)
    echo "unknown model: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${HF_CACHE_ROOT}/${MODEL_SNAPSHOT}/config.json" ]]; then
  echo "missing pinned model snapshot: ${HF_CACHE_ROOT}/${MODEL_SNAPSHOT}" >&2
  exit 1
fi

OUTPUT_DIR="${DATA_ROOT}/vqa/benchmarks/omni-compatibility-3/${RUN_ID}"
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "refusing to reuse existing output path: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir "${OUTPUT_DIR}"

exec podman run --rm \
  --network=none \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=keep-groups \
  --security-opt label=disable \
  --shm-size=16g \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/src \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "${DATA_ROOT}:/data:ro" \
  -v "${HF_CACHE_ROOT}:/cache:ro" \
  -v "${REPO_ROOT}:/workspace:ro" \
  -v "${OUTPUT_DIR}:/output:rw" \
  "${RUNTIME_IMAGE}" \
  python -m trec_video_2026.vqa_experiments.omni_probe \
    --model "${MODEL_KEY}" \
    --model-path "/cache/${MODEL_SNAPSHOT}" \
    --contract "/data/${CONTRACT_REL}" \
    --data-root /data \
    --output-dir /output \
    --runtime-image "${RUNTIME_IMAGE}" \
    --runtime-image-digest "${RUNTIME_IMAGE_DIGEST}" \
    --run-id "${RUN_ID}" \
    --expected-contract-sha256 "${EXPECTED_CONTRACT_SHA256}"
