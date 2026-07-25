#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly HF_CLI_VERSION="1.24.0"

RUN_ID="${1:-}"
[[ "$RUN_ID" =~ ^qwen3-omni-compatibility-3-20260724-a[2-9][0-9]*$ ]] \
  || { echo "error: expected an unused a2-or-later run ID" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_LINK="$REPO_ROOT/data"
[[ -L "$DATA_LINK" ]] || { echo "error: data must be a symlink" >&2; exit 2; }
[[ "$(readlink -f "$DATA_LINK")" == "$EXPECTED_DATA_TARGET" ]] \
  || { echo "error: data symlink does not resolve to the canonical dataset" >&2; exit 2; }

OUTPUT_DIR="$REPO_ROOT/data/vqa/benchmarks/qwen3-omni-compatibility-3/$RUN_ID"
REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$RUN_ID"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "error: refusing to overwrite $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

hf_cli() {
  uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

hf_cli buckets list "$REMOTE_URI" --format json >/dev/null
hf_cli buckets sync "$REMOTE_URI" "$OUTPUT_DIR"
(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

dstack run get "$RUN_ID" --json \
  | uv run python -m trec_video_2026.vqa_experiments.dstack_metadata \
      --run-id "$RUN_ID" \
      --output "$OUTPUT_DIR/dstack-run.json"

(
  cd "$OUTPUT_DIR"
  find . -type f ! -name artifact-hashes.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum >artifact-hashes.sha256
)

echo "$OUTPUT_DIR"
