#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"

RUN_ID="${1:-}"
case "$RUN_ID" in
  nemotron3-omni-runner-canary-20260730-a1|nemotron3-omni-runner-canary-20260730-a2|nemotron3-omni-pilot-20260730-a1) ;;
  *) echo "error: unexpected immutable run ID" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_LINK="$REPO_ROOT/data"
[[ -L "$DATA_LINK" ]] || { echo "error: data must be a symlink" >&2; exit 2; }
[[ "$(readlink -f "$DATA_LINK")" == "$EXPECTED_DATA_TARGET" ]] \
  || { echo "error: data symlink target mismatch" >&2; exit 2; }
[[ -z "${HF_TOKEN:-}" ]] \
  || { echo "error: HF_TOKEN must remain unset outside dstack secrets" >&2; exit 2; }

REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$RUN_ID"
OUTPUT_DIR="$DATA_LINK/vqa/benchmarks/nemotron3-omni/$RUN_ID"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "error: refusing to overwrite $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT_DIR")"
TEMP_OUTPUT_DIR="$(mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX")"
trap 'rm -rf "$TEMP_OUTPUT_DIR"' EXIT

uvx_launcher="$(command -v uvx)"
[[ "$uvx_launcher" = /* && -x "$uvx_launcher" ]] \
  || { echo "error: uvx is unavailable" >&2; exit 2; }
hf_cli() {
  "$uvx_launcher" --from "uv==$UV_VERSION" uvx \
    --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

hf_cli buckets list "${REMOTE_URI}/outputs" --format json >/dev/null
hf_cli buckets sync "${REMOTE_URI}/outputs" "$TEMP_OUTPUT_DIR"
(
  cd "$TEMP_OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)
dstack run get "$RUN_ID" --json \
  | PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
      -m trec_video_2026.vqa_experiments.dstack_metadata \
      --run-id "$RUN_ID" --output "$TEMP_OUTPUT_DIR/dstack-run.json"
(
  cd "$TEMP_OUTPUT_DIR"
  find . -type f ! -name artifact-hashes.sha256 -print0 \
    | sort -z | xargs -0 sha256sum >artifact-hashes.sha256
  sha256sum -c artifact-hashes.sha256
)
mv -T --no-clobber "$TEMP_OUTPUT_DIR" "$OUTPUT_DIR"
[[ -d "$OUTPUT_DIR" && ! -e "$TEMP_OUTPUT_DIR" ]] \
  || { echo "error: refusing to overwrite or nest output" >&2; exit 2; }
trap - EXIT
echo "$OUTPUT_DIR"
