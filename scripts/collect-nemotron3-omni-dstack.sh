#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
readonly EXPANDED12_CANARY_C1_SCHEDULER_ID="nemotron3-expanded12-canary-20260730-c1"
readonly EXPANDED12_CANARY_D1_SCHEDULER_ID="nemotron3-expanded12-canary-20260731-d1"
readonly FROZEN30_CANARY_SCHEDULER_ID="nemotron3-f30-canary-20260801-a1"

RUN_ID="${1:-}"
case "$RUN_ID" in
  nemotron3-omni-runner-canary-20260730-a1|nemotron3-omni-runner-canary-20260730-a2|nemotron3-omni-pilot-20260730-a1|nemotron3-omni-codec-sentinel-20260730-b1|nemotron3-omni-pilot-20260730-b1|nemotron3-omni-expanded12-canary-20260730-c1|nemotron3-omni-expanded12-20260730-c1|nemotron3-omni-expanded12-canary-20260731-d1|nemotron3-omni-expanded12-20260731-d1|nemotron3-omni-frozen30-canary-20260801-a1|nemotron3-omni-frozen30-20260801-a1) ;;
  *) echo "error: unexpected immutable run ID" >&2; exit 2 ;;
esac
DSTACK_RUN_NAME="$RUN_ID"
if [[ "$RUN_ID" == "nemotron3-omni-expanded12-canary-20260730-c1" ]]; then
  DSTACK_RUN_NAME="$EXPANDED12_CANARY_C1_SCHEDULER_ID"
fi
if [[ "$RUN_ID" == "nemotron3-omni-expanded12-canary-20260731-d1" ]]; then
  DSTACK_RUN_NAME="$EXPANDED12_CANARY_D1_SCHEDULER_ID"
fi
if [[ "$RUN_ID" == "nemotron3-omni-frozen30-canary-20260801-a1" ]]; then
  DSTACK_RUN_NAME="$FROZEN30_CANARY_SCHEDULER_ID"
fi
PREVIEW_PATH="${2:-}"
case "$RUN_ID" in
  nemotron3-omni-codec-sentinel-20260730-b1|nemotron3-omni-pilot-20260730-b1)
    [[ -f "$PREVIEW_PATH" ]] \
      || { echo "error: preview output is required for b1" >&2; exit 2; }
    ;;
  nemotron3-omni-expanded12-canary-20260730-c1|nemotron3-omni-expanded12-20260730-c1|nemotron3-omni-expanded12-canary-20260731-d1|nemotron3-omni-expanded12-20260731-d1)
    [[ -f "$PREVIEW_PATH" ]] \
      || { echo "error: preview output is required for c1" >&2; exit 2; }
    ;;
  nemotron3-omni-frozen30-canary-20260801-a1|nemotron3-omni-frozen30-20260801-a1)
    [[ -f "$PREVIEW_PATH" ]] \
      || { echo "error: preview output is required for frozen30 a1" >&2; exit 2; }
    ;;
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
TEMP_AUDIT_DIR="$(mktemp -d /tmp/nemotron3-omni-audit.XXXXXX)"
VERIFY_AUDIT_DIR="$(mktemp -d /tmp/nemotron3-omni-audit-verify.XXXXXX)"
cleanup() {
  rm -rf "$TEMP_OUTPUT_DIR" "$TEMP_AUDIT_DIR" "$VERIFY_AUDIT_DIR"
}
trap cleanup EXIT

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
if [[ "$RUN_ID" == "nemotron3-omni-frozen30-20260801-a1" ]]; then
  [[ "$(find "$TEMP_OUTPUT_DIR/normalized" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 30 ]] \
    || { echo "error: collected bundle must preserve exactly 30 normalized artifacts" >&2; exit 2; }
fi
if [[ -n "$PREVIEW_PATH" ]]; then
  cp -- "$PREVIEW_PATH" "$TEMP_AUDIT_DIR/preview.txt"
  dstack run get "$DSTACK_RUN_NAME" --json \
    | PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
        -m trec_video_2026.vqa_experiments.dstack_metadata \
        --run-id "$DSTACK_RUN_NAME" --output "$TEMP_AUDIT_DIR/dstack-run.json"
  cp -- "$TEMP_AUDIT_DIR/dstack-run.json" "$TEMP_OUTPUT_DIR/dstack-run.json"
  (
    cd "$TEMP_AUDIT_DIR"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum -c SHA256SUMS
    chmod a-w ./*
  )
  audit_listing="$(hf_cli buckets list "${REMOTE_URI}/audit" --format quiet)"
  [[ -z "$audit_listing" ]] \
    || { echo "error: immutable audit prefix already exists" >&2; exit 2; }
  hf_cli buckets sync "$TEMP_AUDIT_DIR" "${REMOTE_URI}/audit"
  hf_cli buckets list "${REMOTE_URI}/audit" --format json
  hf_cli buckets sync "${REMOTE_URI}/audit" "$VERIFY_AUDIT_DIR"
  (
    cd "$VERIFY_AUDIT_DIR"
    sha256sum -c SHA256SUMS
  )
else
  dstack run get "$DSTACK_RUN_NAME" --json \
  | PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
      -m trec_video_2026.vqa_experiments.dstack_metadata \
      --run-id "$DSTACK_RUN_NAME" --output "$TEMP_OUTPUT_DIR/dstack-run.json"
fi
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
rm -rf "$TEMP_AUDIT_DIR" "$VERIFY_AUDIT_DIR"
echo "$OUTPUT_DIR"
