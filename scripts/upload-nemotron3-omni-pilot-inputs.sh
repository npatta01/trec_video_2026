#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly EXPECTED_RUN_ID_A1="nemotron3-omni-pilot-20260730-a1"
readonly EXPECTED_RUN_ID_B1="nemotron3-omni-pilot-20260730-b1"
readonly EXPECTED_RUN_ID_C1="nemotron3-omni-expanded12-20260730-c1"
readonly EXPECTED_RUN_ID_D1="nemotron3-omni-expanded12-20260731-d1"
readonly EXPECTED_RUN_ID_FROZEN30="nemotron3-omni-frozen30-20260801-a1"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"

RUN_ID="${1:-$EXPECTED_RUN_ID_A1}"
case "$RUN_ID" in
  "$EXPECTED_RUN_ID_A1"|"$EXPECTED_RUN_ID_B1"|"$EXPECTED_RUN_ID_C1"|"$EXPECTED_RUN_ID_D1"|"$EXPECTED_RUN_ID_FROZEN30") ;;
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

REMOTE_URI="hf://buckets/Npatta01/trec_video_2026/experiments/$RUN_ID"
media_specs=(
  "IRTTtnSRTyY.mkv:vqa/2025/training/videos_original/IRTTtnSRTyY.mkv"
  "IWSPAfCKtV8.mkv:vqa/2025/training/videos_original/IWSPAfCKtV8.mkv"
  "MdMKHhFFqJQ.mkv:vqa/2025/training/videos_original/MdMKHhFFqJQ.mkv"
  "OzNdzgQi1CA.mkv:vqa/2025/training/videos_original/OzNdzgQi1CA.mkv"
  "EMI_anpHII4.mkv:vqa/2025/training/videos_original/EMI_anpHII4.mkv"
  "LusRurRrmf0.mkv:vqa/2025/training/videos_original/LusRurRrmf0.mkv"
)
if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_C1" || "$RUN_ID" == "$EXPECTED_RUN_ID_D1" ]]; then
  media_specs=(
    "NN7G28sBOlY.mkv:vqa/2025/training/videos_original/NN7G28sBOlY.mkv"
    "NUmb2P3Bzu8.mkv:vqa/2025/training/videos_original/NUmb2P3Bzu8.mkv"
    "IRTTtnSRTyY.mkv:vqa/2025/training/videos_original/IRTTtnSRTyY.mkv"
    "IWSPAfCKtV8.mkv:vqa/2025/training/videos_original/IWSPAfCKtV8.mkv"
    "MdMKHhFFqJQ.mkv:vqa/2025/training/videos_original/MdMKHhFFqJQ.mkv"
    "OzNdzgQi1CA.mkv:vqa/2025/training/videos_original/OzNdzgQi1CA.mkv"
    "CH8Dp7EgW4I.mkv:vqa/2025/training/videos_original/CH8Dp7EgW4I.mkv"
    "EtNOkUJd5w8.mkv:vqa/2025/training/videos_original/EtNOkUJd5w8.mkv"
    "Gp4LVTn3YbY.mkv:vqa/2025/training/videos_original/Gp4LVTn3YbY.mkv"
    "KgOvCPOX-Ms.mkv:vqa/2025/training/videos_original/KgOvCPOX-Ms.mkv"
    "EMI_anpHII4.mkv:vqa/2025/training/videos_original/EMI_anpHII4.mkv"
    "LusRurRrmf0.mkv:vqa/2025/training/videos_original/LusRurRrmf0.mkv"
  )
fi
readonly media_specs

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
contract_command="build-contract"
input_count=7
panel="$REPO_ROOT/configs/vqa/failure_panel_12.json"
if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_C1" || "$RUN_ID" == "$EXPECTED_RUN_ID_D1" ]]; then
  contract_command="build-expanded-contract"
  input_count=13
fi
if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_FROZEN30" ]]; then
  contract_command="build-frozen30-contract"
  input_count=31
  panel="$REPO_ROOT/configs/vqa/native_audio_benchmark_30_inference.json"
fi
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
  -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot "$contract_command" \
  --panel "$panel" \
  --data-root "$DATA_LINK" >"$contract"

if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_FROZEN30" ]]; then
  mapfile -d '' -t frozen30_media < <(
    "$PYTHON_BIN" - "$contract" <<'PY'
import json
import sys
from pathlib import PurePosixPath

contract = json.load(open(sys.argv[1], encoding="utf-8"))
examples = contract.get("examples")
if contract.get("mode") != "frozen30-paired" or not isinstance(examples, list) or len(examples) != 30:
    raise RuntimeError("frozen30 contract must declare exactly 30 media")
paths = []
for example in examples:
    relative = example.get("media_path") if isinstance(example, dict) else None
    path = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
    if path.is_absolute() or ".." in path.parts or path.as_posix() in paths:
        raise RuntimeError("frozen30 contract media paths must be unique and relative")
    paths.append(path.as_posix())
for path in paths:
    sys.stdout.buffer.write(path.encode() + b"\0")
PY
  )
  [[ "${#frozen30_media[@]}" -eq 30 ]] \
    || { echo "error: frozen30 contract did not yield exactly 30 media" >&2; exit 2; }

  "$PYTHON_BIN" - "$contract" "$marker" "$DATA_LINK" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

contract_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
data_root = Path(sys.argv[3]).resolve(strict=True)
contract = json.loads(contract_path.read_text(encoding="utf-8"))
objects = {"contract.json": hashlib.sha256(contract_path.read_bytes()).hexdigest()}
for example in contract["examples"]:
    relative = PurePosixPath(example["media_path"])
    source = data_root.joinpath(*relative.parts).resolve(strict=True)
    source.relative_to(data_root)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != example["media_sha256"]:
        raise RuntimeError(f"contract media hash mismatch: {relative}")
    objects[relative.as_posix()] = digest
if len(objects) != 31:
    raise RuntimeError("frozen30 input bundle must contain 31 content objects")
marker_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "input_count": 31,
            "object_count": 32,
            "contract_sha256": objects["contract.json"],
            "objects": objects,
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

  hf_cli buckets cp "$contract" "$REMOTE_URI/inputs/contract.json"
  for relative in "${frozen30_media[@]}"; do
    hf_cli buckets cp "$DATA_LINK/$relative" "$REMOTE_URI/inputs/$relative"
  done
  hf_cli buckets cp "$marker" "$REMOTE_URI/inputs/INPUTS_COMPLETE.json"
  hf_cli buckets list "$REMOTE_URI/inputs" --recursive --format json
  verify_dir="$temp_dir/verify"
  mkdir -p "$verify_dir"
  hf_cli buckets sync "$REMOTE_URI/inputs" "$verify_dir"
  "$PYTHON_BIN" - "$verify_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
marker_path = root / "INPUTS_COMPLETE.json"
marker = json.loads(marker_path.read_text(encoding="utf-8"))
if set(marker) != {"schema_version", "input_count", "object_count", "contract_sha256", "objects"}:
    raise RuntimeError("frozen30 input marker schema mismatch")
objects = marker.get("objects")
if (
    marker.get("schema_version") != 1
    or marker.get("input_count") != 31
    or marker.get("object_count") != 32
    or not isinstance(objects, dict)
    or len(objects) != 31
):
    raise RuntimeError("frozen30 input marker identity mismatch")
actual = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
)
expected = sorted(["INPUTS_COMPLETE.json", *objects])
if actual != expected:
    raise RuntimeError("frozen30 remote input object set mismatch")
for relative, expected_digest in objects.items():
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if digest != expected_digest:
        raise RuntimeError(f"frozen30 remote input hash mismatch: {relative}")
if marker["contract_sha256"] != objects["contract.json"]:
    raise RuntimeError("frozen30 marker contract hash mismatch")
PY
  exit 0
fi

hf_cli buckets cp "$contract" "$REMOTE_URI/inputs/contract.json"
for spec in "${media_specs[@]}"; do
  filename="${spec%%:*}"
  relative="${spec#*:}"
  hf_cli buckets cp "$DATA_LINK/$relative" "$REMOTE_URI/inputs/media/$filename"
done
"$PYTHON_BIN" - "$contract" "$marker" "$input_count" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = Path(sys.argv[1])
marker = Path(sys.argv[2])
input_count = int(sys.argv[3])
marker.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "input_count": input_count,
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
