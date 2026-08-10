#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly MODEL_ID="Qwen/Qwen3.5-4B"
readonly MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
readonly WHISPER_MODEL_ID="openai/whisper-large-v3"
readonly WHISPER_REVISION="06f233fe06e710322aca913c1bc4249a0d71fce1"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_VERSION="5.15.0.dev0"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.26.0"
readonly PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
readonly INPUT_DIR="${INPUT_DIR:-/dstack/input}"
readonly FRAME_DIR="${FRAME_DIR:-/dstack/frames}"
readonly AUDIO_DIR="${AUDIO_DIR:-/dstack/audio}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
readonly PROFILE_PATH="${PROFILE:-/dstack/run/configs/vqa/qwen35_whisper_audit_full_profile.json}"

COUNT=""
FRAME_COUNT=""
MANIFEST=""

die() {
  echo "error: $*" >&2
  exit 2
}

hf_cli() {
  HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}" \
    uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

load_profile() {
  [[ -x "$PYTHON_BIN" ]] || die "missing image interpreter: $PYTHON_BIN"
  [[ -f "$PROFILE_PATH" ]] || die "missing run profile: $PROFILE_PATH"
  mkdir -p "$INPUT_DIR" "$FRAME_DIR" "$AUDIO_DIR" "$OUTPUT_DIR"
  "$PYTHON_BIN" - "$PROFILE_PATH" "$OUTPUT_DIR/profile-preflight.json" \
    "$EXPECTED_RUNTIME_IMAGE" "${EXPERIMENT_RUN_NAME:-}" "${HF_INPUT_URI:-}" \
    "${HF_REMOTE_URI:-}" <<'PY'
import json
import sys
from pathlib import Path

profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_image, env_name, env_input, env_remote = sys.argv[3:7]
required = {"run_name", "input_uri", "remote_uri", "manifest", "count", "frame_count"}
if set(profile) != required:
    raise SystemExit(f"profile fields must be exactly {sorted(required)}")
if profile["run_name"] != env_name:
    raise SystemExit("profile run name does not match environment")
if profile["input_uri"] != env_input:
    raise SystemExit("profile input URI does not match environment")
if profile["remote_uri"] != env_remote:
    raise SystemExit("profile remote URI does not match environment")
if profile["count"] not in {3, 475} or profile["frame_count"] != 12:
    raise SystemExit("profile count/frame_count is not an approved value")
manifest = Path(profile["manifest"])
if not manifest.is_file():
    raise SystemExit(f"missing manifest: {manifest}")
manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
if manifest_payload.get("label_policy") != "label-free-inference-input":
    raise SystemExit("manifest is not label-free")
examples = manifest_payload.get("examples")
if not isinstance(examples, list) or len(examples) != profile["count"]:
    raise SystemExit("manifest count does not match profile")
for index, example in enumerate(examples):
    if set(example) != {"video_id", "question"}:
        raise SystemExit(f"manifest row {index} contains unexpected fields")
Path(sys.argv[2]).write_text(
    json.dumps({
        "run_name": profile["run_name"],
        "input_uri": profile["input_uri"],
        "remote_uri": profile["remote_uri"],
        "count": profile["count"],
        "frame_count": profile["frame_count"],
        "runtime_image": expected_image,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  COUNT="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["count"])' "$PROFILE_PATH")"
  FRAME_COUNT="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["frame_count"])' "$PROFILE_PATH")"
  MANIFEST="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest"])' "$PROFILE_PATH")"
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "uv==$UV_VERSION"
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ https://github.com/huggingface/transformers/archive/$TRANSFORMERS_REVISION.tar.gz" \
    "qwen-vl-utils[decord]==0.0.14" \
    "av==16.0.1" \
    "accelerate==1.12.0" \
    "imageio-ffmpeg==0.6.0" \
    "Pillow==11.3.0" \
    "soundfile==0.13.1"
  "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" <<'PY'
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig

if transformers.__version__ != "5.15.0.dev0":
    raise SystemExit(f"unexpected transformers version: {transformers.__version__}")
if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip:
    raise SystemExit("NVIDIA CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bfloat16 is unavailable")
config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
record = {
    "model_type": config.model_type,
    "transformers_version": transformers.__version__,
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_bf16_supported": True,
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY
}

remote_prefix_is_empty() {
  local listing list_rc
  set +e
  listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format quiet 2>"$OUTPUT_DIR/remote-list.stderr")"
  list_rc=$?
  set -e
  if (( list_rc != 0 )); then
    sed -n '1,120p' "$OUTPUT_DIR/remote-list.stderr" >&2
    die "could not inspect remote output prefix (hf rc=$list_rc)"
  fi
  [[ -z "$listing" ]] || die "remote output prefix already exists: $HF_REMOTE_URI"
}

verify_input_bundle() {
  "$PYTHON_BIN" - "$INPUT_DIR" "$OUTPUT_DIR/input-preflight.json" "$COUNT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve(strict=True)
expected_count = int(sys.argv[3])
contract_path = root / "contract.json"
marker_path = root / "INPUTS_COMPLETE.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))
marker = json.loads(marker_path.read_text(encoding="utf-8"))
objects = contract.get("objects")
if contract.get("schema_version") != 1 or contract.get("media_count") != 475:
    raise SystemExit("input contract identity mismatch")
if marker.get("schema_version") != 1 or marker.get("input_count") != 475:
    raise SystemExit("input marker identity mismatch")
if marker.get("object_count") != 477 or marker.get("media_count") != 475:
    raise SystemExit("input marker object count mismatch")
if not isinstance(objects, list) or len(objects) != 475:
    raise SystemExit("input contract must contain 475 media objects")
expected_paths = {"contract.json", "INPUTS_COMPLETE.json"}
media_bytes = 0
for record in objects:
    relative_raw = record.get("path")
    expected_digest = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(relative_raw, str) or not isinstance(expected_digest, str):
        raise SystemExit("input contract has invalid object identity")
    relative = PurePosixPath(relative_raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe input path: {relative}")
    path = root.joinpath(*relative.parts).resolve(strict=True)
    if not path.is_relative_to(root):
        raise SystemExit(f"input path escapes root: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest or path.stat().st_size != expected_size:
        raise SystemExit(f"input media hash/size mismatch: {relative}")
    expected_paths.add(relative.as_posix())
    media_bytes += path.stat().st_size
actual_paths = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
}
if actual_paths != expected_paths:
    raise SystemExit(
        f"input object set mismatch: missing={sorted(expected_paths - actual_paths)} "
        f"extra={sorted(actual_paths - expected_paths)}"
    )
if marker.get("contract_sha256") != hashlib.sha256(contract_path.read_bytes()).hexdigest():
    raise SystemExit("input contract hash does not match marker")
if marker.get("media_bytes") != media_bytes:
    raise SystemExit("input media byte count does not match marker")
Path(sys.argv[2]).write_text(json.dumps({
    "profile_count": expected_count,
    "contract_sha256": marker["contract_sha256"],
    "media_count": 475,
    "media_bytes": media_bytes,
    "object_count": len(actual_paths),
}, indent=2, sort_keys=True) + "\n")
PY
}

prepare_evidence() {
  local started_ns finished_ns
  started_ns="$(date +%s%N)"
  hf_cli buckets sync "$HF_INPUT_URI" "$INPUT_DIR"
  verify_input_bundle
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_prepare_matched_evidence.py \
    --manifest "$MANIFEST" \
    --input-root "$INPUT_DIR" \
    --frame-root "$FRAME_DIR" \
    --audio-root "$AUDIO_DIR" \
    --contact-sheet-root "$OUTPUT_DIR/frame-contact-sheets" \
    --requests-output "$OUTPUT_DIR/requests.frames-only.json" \
    --evidence-output "$OUTPUT_DIR/frame-evidence.json" \
    --count "$COUNT" \
    --frame-count "$FRAME_COUNT" \
    --workers 4 \
    --run-id "$EXPERIMENT_RUN_NAME" \
    > "$OUTPUT_DIR/prepare.stdout" 2> "$OUTPUT_DIR/prepare.stderr"
  finished_ns="$(date +%s%N)"
  "$PYTHON_BIN" - "$OUTPUT_DIR/prepare-summary.json" "$started_ns" "$finished_ns" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "prepare_wall_seconds": (int(sys.argv[3]) - int(sys.argv[2])) / 1_000_000_000,
    "count": len(json.loads(Path("/dstack/output/requests.frames-only.json").read_text())),
}, indent=2, sort_keys=True) + "\n")
PY
}

run_whisper() {
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_whisper_cuda.py \
    --evidence "$OUTPUT_DIR/frame-evidence.json" \
    --output "$OUTPUT_DIR/whisper-transcripts.json" \
    --summary "$OUTPUT_DIR/whisper-summary.json" \
    --model-id "$WHISPER_MODEL_ID" \
    --revision "$WHISPER_REVISION" \
    --count "$COUNT" \
    > "$OUTPUT_DIR/whisper.stdout" 2> "$OUTPUT_DIR/whisper.stderr"
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_build_whisper_requests.py \
    --frames "$OUTPUT_DIR/requests.frames-only.json" \
    --whisper "$OUTPUT_DIR/whisper-transcripts.json" \
    --output "$OUTPUT_DIR/requests.frames-whisper.json" \
    --condition-id qwen35-4b-12f-whisper-large-v3 \
    --count "$COUNT" \
    > "$OUTPUT_DIR/whisper-request-build.stdout" \
    2> "$OUTPUT_DIR/whisper-request-build.stderr"
}

run_condition() {
  local condition_id="$1"
  local request_path="$2"
  local prediction_path="$3"
  local stderr_path="$4"
  local summary_path="$5"
  local started_ns finished_ns child_rc
  started_ns="$(date +%s%N)"
  set +e
  "$PYTHON_BIN" -m trec_video_2026.vqa_transformers_runner \
    --model-id "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --model-class image-text \
    --max-new-tokens 64 \
    < "$request_path" \
    > "$prediction_path" \
    2> "$stderr_path"
  child_rc=$?
  set -e
  finished_ns="$(date +%s%N)"
  "$PYTHON_BIN" - "$summary_path" "$condition_id" "$child_rc" "$started_ns" "$finished_ns" "$prediction_path" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "condition_id": sys.argv[2],
    "workload_returncode": int(sys.argv[3]),
    "wall_seconds": (int(sys.argv[5]) - int(sys.argv[4])) / 1_000_000_000,
}
prediction_path = Path(sys.argv[6])
if prediction_path.is_file():
    try:
        rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        rows = None
    if isinstance(rows, list):
        payload["rows"] = len(rows)
        payload["statuses"] = {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status") for row in rows})
        }
        payload["devices"] = sorted({row.get("device") for row in rows})
        payload["model_load_seconds"] = sorted({row.get("model_load_seconds") for row in rows})
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  return "$child_rc"
}

validate_predictions() {
  local request_path="$1" prediction_path="$2" expected_condition="$3"
  "$PYTHON_BIN" - "$request_path" "$prediction_path" "$expected_condition" "$COUNT" <<'PY'
import json
import sys
from pathlib import Path
requests = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
predictions = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_condition = sys.argv[3]
expected_count = int(sys.argv[4])
if not isinstance(predictions, list) or len(predictions) != expected_count:
    raise SystemExit("prediction count mismatch")
if [row.get("video_id") for row in predictions] != [row["video_id"] for row in requests]:
    raise SystemExit("prediction IDs/order do not match requests")
if any(row.get("condition_id") != expected_condition for row in predictions):
    raise SystemExit("prediction condition mismatch")
if any(row.get("status") != "passed" for row in predictions):
    raise SystemExit("one or more prediction rows did not pass")
if any(not isinstance(row.get("answer"), str) or not row["answer"].strip() for row in predictions):
    raise SystemExit("one or more predictions has an empty answer")
PY
}

validate_whisper() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/whisper-transcripts.json" "$COUNT" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload.get("results")
if payload.get("label_policy") != "label-free-transcript-evidence":
    raise SystemExit("Whisper label policy mismatch")
if not isinstance(rows, list) or len(rows) != int(sys.argv[2]):
    raise SystemExit("Whisper row count mismatch")
if any(row.get("status") not in {"passed", "no-audio-stream"} for row in rows):
    raise SystemExit("Whisper has failed rows")
PY
}

finalize() {
  local workload_rc="$1" sync_rc
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"workload_returncode": int(sys.argv[2])}, sort_keys=True) + "\n")
PY
  (cd "$OUTPUT_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI/output"
  sync_rc=$?
  set -e
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" "$sync_rc" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "workload_returncode": int(sys.argv[2]),
    "sync_returncode": int(sys.argv[3]),
}, sort_keys=True) + "\n")
PY
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

main() {
  local workload_rc=0
  load_profile
  export COUNT FRAME_COUNT MANIFEST EXPERIMENT_RUN_NAME HF_INPUT_URI HF_REMOTE_URI
  bootstrap_runtime
  command -v uvx >/dev/null 2>&1 || die "uvx is missing after pinned uv installation"
  uvx hf version > "$OUTPUT_DIR/hf-version.txt"
  uvx hf auth whoami > "$OUTPUT_DIR/hf-whoami.txt"
  remote_prefix_is_empty
  set +e
  prepare_evidence
  workload_rc=$?
  set -e
  if (( workload_rc == 0 )); then
    set +e
    run_whisper
    workload_rc=$?
    set -e
  fi
  if (( workload_rc == 0 )); then
    set +e
    validate_whisper
    workload_rc=$?
    set -e
  fi
  if (( workload_rc == 0 )); then
    set +e
    run_condition qwen35-4b-12f-frames-only \
      "$OUTPUT_DIR/requests.frames-only.json" \
      "$OUTPUT_DIR/predictions.frames-only.json" \
      "$OUTPUT_DIR/inference.frames-only.stderr" \
      "$OUTPUT_DIR/summary.frames-only.json"
    workload_rc=$?
    set -e
  fi
  if (( workload_rc == 0 )); then
    set +e
    run_condition qwen35-4b-12f-whisper-large-v3 \
      "$OUTPUT_DIR/requests.frames-whisper.json" \
      "$OUTPUT_DIR/predictions.frames-whisper.json" \
      "$OUTPUT_DIR/inference.frames-whisper.stderr" \
      "$OUTPUT_DIR/summary.frames-whisper.json"
    workload_rc=$?
    set -e
  fi
  if (( workload_rc == 0 )); then
    set +e
    validate_predictions "$OUTPUT_DIR/requests.frames-only.json" "$OUTPUT_DIR/predictions.frames-only.json" qwen35-4b-12f-frames-only
    workload_rc=$?
    set -e
  fi
  if (( workload_rc == 0 )); then
    set +e
    validate_predictions "$OUTPUT_DIR/requests.frames-whisper.json" "$OUTPUT_DIR/predictions.frames-whisper.json" qwen35-4b-12f-whisper-large-v3
    workload_rc=$?
    set -e
  fi
  finalize "$workload_rc"
}

main "$@"
