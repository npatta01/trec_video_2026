#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 REQUIREMENTS" >&2
  exit 2
fi

readonly REQUIREMENTS_PATH=$1
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip uninstall --break-system-packages --yes spin
"${PYTHON_BIN}" -m pip install --break-system-packages --no-cache-dir --no-deps \
  "uv==0.11.3"
"${PYTHON_BIN}" -m uv pip install --system --break-system-packages \
  --no-managed-python --requirements "${REQUIREMENTS_PATH}"
"${PYTHON_BIN}" -m uv pip check --system
