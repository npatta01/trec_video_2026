#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly EXPECTED_RUN_ID="qwen3-omni-quality-10-20260725-a1"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
UVX_BIN=""

RUN_ID="${1:-}"
[[ "$RUN_ID" == "$EXPECTED_RUN_ID" ]] \
  || { echo "error: expected immutable run ID $EXPECTED_RUN_ID" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/qwen3-omni-quality-collector-lib.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_LINK="$REPO_ROOT/data"
[[ -L "$DATA_LINK" ]] || { echo "error: data must be a symlink" >&2; exit 2; }
[[ "$(readlink -f "$DATA_LINK")" == "$EXPECTED_DATA_TARGET" ]] \
  || { echo "error: data symlink does not resolve to the canonical dataset" >&2; exit 2; }

OUTPUT_DIR="$REPO_ROOT/data/vqa/benchmarks/qwen3-omni-quality-10/$RUN_ID"
REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$RUN_ID"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "error: refusing to overwrite $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT_DIR")"
TEMP_OUTPUT_DIR="$(mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX")"
trap 'rm -rf "$TEMP_OUTPUT_DIR"' EXIT

hf_cli() {
  "$UVX_BIN" --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

bootstrap_uv() {
  [[ -x "$PYTHON_BIN" ]] || { echo "error: missing exact Python interpreter" >&2; exit 2; }
  "$PYTHON_BIN" -m pip install --no-deps "uv==$UV_VERSION"
  [[ "$("$PYTHON_BIN" -c \
    'import importlib.metadata; print(importlib.metadata.version("uv"))')" == "$UV_VERSION" ]] \
    || { echo "error: uv bootstrap attestation failed" >&2; exit 2; }
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
    echo "error: could not resolve the pinned uvx entry point" >&2
    exit 2
  fi
  [[ "$UVX_BIN" = /* && -x "$UVX_BIN" ]] \
    || { echo "error: pinned uvx entry point is unavailable" >&2; exit 2; }
}

bootstrap_uv
hf_cli buckets list "$REMOTE_URI" --format json >/dev/null
hf_cli buckets sync "$REMOTE_URI" "$TEMP_OUTPUT_DIR"
(
  cd "$TEMP_OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

dstack run get "$RUN_ID" --json \
  | PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.dstack_metadata \
      --run-id "$RUN_ID" \
      --output "$TEMP_OUTPUT_DIR/dstack-run.json"

(
  cd "$TEMP_OUTPUT_DIR"
  find . -type f ! -name artifact-hashes.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum >artifact-hashes.sha256
  sha256sum -c artifact-hashes.sha256
)

publish_verified_output "$TEMP_OUTPUT_DIR" "$OUTPUT_DIR"
trap - EXIT
echo "$OUTPUT_DIR"
