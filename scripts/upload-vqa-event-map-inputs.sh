#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_video_2026/experiments/vqa-event-map-inputs-20260809-a3/inputs"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_LINK="$REPO_ROOT/data"
MANIFEST="$REPO_ROOT/configs/vqa/qwen35_vqa2025_475_label_free.json"
EVIDENCE="$DATA_LINK/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/frame-evidence.json"
MEDIA_ROOT="$DATA_LINK/vqa/2025/training/videos_original"
BUNDLE_DIR="$DATA_LINK/vqa/benchmarks/vqa-event-map-inputs-20260809-a3/inputs"

[[ -z "${HF_TOKEN:-}" ]] \
  || { echo "error: HF_TOKEN must remain unset outside dstack secrets" >&2; exit 2; }
[[ -L "$DATA_LINK" ]] || { echo "error: data must be a symlink" >&2; exit 2; }
[[ "$(readlink -f "$DATA_LINK")" == "$EXPECTED_DATA_TARGET" ]] \
  || { echo "error: data symlink target mismatch" >&2; exit 2; }
[[ -f "$MANIFEST" && -f "$EVIDENCE" && -d "$MEDIA_ROOT" ]] \
  || { echo "error: canonical label-free inputs are unavailable" >&2; exit 2; }

UVX_LAUNCHER="$(command -v uvx)"
[[ "$UVX_LAUNCHER" = /* && -x "$UVX_LAUNCHER" ]] \
  || { echo "error: uvx is unavailable" >&2; exit 2; }

hf_cli() {
  "$UVX_LAUNCHER" --from "uv==$UV_VERSION" uvx \
    --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

if [[ ! -d "$BUNDLE_DIR" ]]; then
  mkdir -p "$(dirname "$BUNDLE_DIR")"
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
    "$REPO_ROOT/scripts/build_vqa_event_map_bundle.py" \
    --manifest "$MANIFEST" \
    --evidence "$EVIDENCE" \
    --media-root "$MEDIA_ROOT" \
    --output-dir "$BUNDLE_DIR" >/dev/null
fi

(
  cd "$BUNDLE_DIR"
  sha256sum -c SHA256SUMS >/dev/null
)

mapfile -d '' -t MEDIA_NAMES < <(
  "$PYTHON_BIN" - "$BUNDLE_DIR/event-map-controller.json" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath

controller = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = controller.get("rows")
if (
    controller.get("schema_version") != 1
    or controller.get("label_policy") != "label-free-controller"
    or not isinstance(rows, list)
    or len(rows) != 475
):
    raise SystemExit("controller must contain the exact 475-row label-free panel")
names = []
for row in rows:
    relative = PurePosixPath(row.get("media_path", ""))
    if relative.parts[:1] != ("media",) or len(relative.parts) != 2:
        raise SystemExit("controller media path is unsafe")
    names.append(relative.name)
if len(set(names)) != 475:
    raise SystemExit("controller media names must be unique")
for name in names:
    sys.stdout.buffer.write(name.encode("utf-8") + b"\0")
PY
)
[[ "${#MEDIA_NAMES[@]}" -eq 475 ]] \
  || { echo "error: exact 475-video upload set is unavailable" >&2; exit 2; }

REMOTE_LISTING="$(hf_cli buckets list "$EXPECTED_INPUT_URI" --recursive --format quiet)"
[[ -z "$REMOTE_LISTING" ]] \
  || { echo "error: immutable input prefix already exists" >&2; exit 2; }

# Metadata is uploaded first, media bytes are streamed directly from the
# canonical dataset, and SHA256SUMS is the terminal input marker.
hf_cli buckets sync "$BUNDLE_DIR" "$EXPECTED_INPUT_URI" \
  --exclude SHA256SUMS >/dev/null

MEDIA_FILTERS=()
for media_name in "${MEDIA_NAMES[@]}"; do
  MEDIA_FILTERS+=(--include "$media_name")
done
hf_cli buckets sync "$MEDIA_ROOT" "$EXPECTED_INPUT_URI/media" \
  "${MEDIA_FILTERS[@]}" >/dev/null
REMOTE_MEDIA_COUNT="$(
  hf_cli buckets list "$EXPECTED_INPUT_URI/media" --recursive --format quiet \
    | wc -l
)"
[[ "$REMOTE_MEDIA_COUNT" -eq 475 ]] \
  || { echo "error: remote input prefix does not contain exactly 475 media" >&2; exit 2; }
hf_cli buckets cp "$BUNDLE_DIR/SHA256SUMS" \
  "$EXPECTED_INPUT_URI/SHA256SUMS" >/dev/null

VERIFY_DIR="$(mktemp -d /tmp/vqa-event-map-input-verify.XXXXXX)"
cleanup() {
  if [[ "$VERIFY_DIR" == /tmp/vqa-event-map-input-verify.* && -d "$VERIFY_DIR" ]]; then
    rm -rf -- "$VERIFY_DIR"
  fi
}
trap cleanup EXIT
hf_cli buckets sync "$EXPECTED_INPUT_URI" "$VERIFY_DIR" \
  --exclude "media/**" >/dev/null
(
  cd "$VERIFY_DIR"
  sha256sum -c SHA256SUMS >/dev/null
)

"$PYTHON_BIN" - "$BUNDLE_DIR/event-map-summary.json" "$EXPECTED_INPUT_URI" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "status": "uploaded",
    "row_count": summary["row_count"],
    "ids_sha256": summary["ids_sha256"],
    "protocol_sha256": summary["protocol_sha256"],
    "input_uri": sys.argv[2],
}, sort_keys=True))
PY
