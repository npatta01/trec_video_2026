#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_LINK="$REPO_ROOT/data"
[[ -L "$DATA_LINK" ]] || { echo "error: data must be a symlink" >&2; exit 2; }
[[ "$(readlink -f "$DATA_LINK")" == "$EXPECTED_DATA_TARGET" ]] \
  || { echo "error: data symlink does not resolve to the canonical dataset" >&2; exit 2; }

cd "$REPO_ROOT"
uv run python -m trec_video_2026.vqa_experiments.qwen3_omni_quality_transport \
  prepare \
  --data-root "$DATA_LINK"
