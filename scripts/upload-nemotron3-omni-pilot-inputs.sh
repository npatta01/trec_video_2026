#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly EXPECTED_RUN_ID_A1="nemotron3-omni-pilot-20260730-a1"
readonly EXPECTED_RUN_ID_B1="nemotron3-omni-pilot-20260730-b1"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"

RUN_ID="${1:-$EXPECTED_RUN_ID_A1}"
case "$RUN_ID" in
  "$EXPECTED_RUN_ID_A1"|"$EXPECTED_RUN_ID_B1") ;;
  *) echo "error: unexpected immutable run ID" >&2; exit 2 ;;
esac
[[ -z "${HF_TOKEN:-}" ]] \
  || { echo "error: HF_TOKEN must remain unset outside dstack secrets" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_LINK="$REPO_ROOT/data"
[[ -L "$DATA_LINK" ]] || { echo "error: data must be a symlink" >&2; exit 2; }
[[ "$(readlink -f "$DATA_LINK")" == "$EXPECTED_DATA_TARGET" ]] \
  || { echo "error: data symlink target mismatch" >&2; exit 2; }

REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$RUN_ID"
readonly media_specs=(
  "IRTTtnSRTyY.mkv:vqa/2025/training/videos_original/IRTTtnSRTyY.mkv"
  "IWSPAfCKtV8.mkv:vqa/2025/training/videos_original/IWSPAfCKtV8.mkv"
  "MdMKHhFFqJQ.mkv:vqa/2025/training/videos_original/MdMKHhFFqJQ.mkv"
  "OzNdzgQi1CA.mkv:vqa/2025/training/videos_original/OzNdzgQi1CA.mkv"
  "EMI_anpHII4.mkv:vqa/2025/training/videos_original/EMI_anpHII4.mkv"
  "LusRurRrmf0.mkv:vqa/2025/training/videos_original/LusRurRrmf0.mkv"
)

uvx_launcher="$(command -v uvx)"
[[ "$uvx_launcher" = /* && -x "$uvx_launcher" ]] \
  || { echo "error: uvx is unavailable" >&2; exit 2; }

hf_cli() {
  "$uvx_launcher" --from "uv==$UV_VERSION" uvx \
    --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

listing="$(hf_cli buckets list "$REMOTE_URI" --format quiet)"
[[ -z "$listing" ]] || { echo "error: immutable run prefix already exists" >&2; exit 2; }

temp_dir="$(mktemp -d /tmp/nemotron3-omni-inputs.XXXXXX)"
trap 'rm -rf "$temp_dir"' EXIT
contract="$temp_dir/contract.json"
marker="$temp_dir/INPUTS_COMPLETE.json"
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
  -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot build-contract \
  --panel "$REPO_ROOT/configs/vqa/failure_panel_12.json" \
  --data-root "$DATA_LINK" >"$contract"

hf_cli buckets cp "$contract" "$REMOTE_URI/inputs/contract.json"
for spec in "${media_specs[@]}"; do
  filename="${spec%%:*}"
  relative="${spec#*:}"
  hf_cli buckets cp "$DATA_LINK/$relative" "$REMOTE_URI/inputs/media/$filename"
done
"$PYTHON_BIN" - "$contract" "$marker" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = Path(sys.argv[1])
marker = Path(sys.argv[2])
marker.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "input_count": 7,
            "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
hf_cli buckets cp "$marker" "$REMOTE_URI/inputs/INPUTS_COMPLETE.json"
hf_cli buckets list "$REMOTE_URI/inputs" --format json
